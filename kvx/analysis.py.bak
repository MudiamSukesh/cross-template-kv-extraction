"""Failure analysis: WHICH structural assumptions break on unseen templates, and how much.

Part A - stratified link recall (link head on gold entities), seen_test vs unseen_test, broken
         down by structural property of each gold key->value pair:
           direction        value right / below / left / above / diagonal of its key
           distance         edge gap in median word heights
           value_lines      single-line vs multi-line value
           key_fanout       key owns one value vs several (table / list-like)
           value_type       numeric / date-like / alphabetic / mixed
           source           dataset
Part B - error types for gold-linked values: correct, wrong key (and whether the chosen key was
         NEARER than the gold key - a nearest-key bias), or "no key" predicted.
Part C - tagging confusion (gold role vs predicted role, word level).
Part D - counterfactual probes: re-run the link head on edited copies of the pages and measure
         how value->key accuracy changes. Each probe tests one assumption:
           shift_scale      global translate + scale             (should not matter: relative geometry)
           mirror_x         flip the page left-right             (left-to-right reading assumption)
           value_below      move each value directly under its key   (right-of-key assumption)
           value_right      move each value directly right of its key (below-key assumption)
           key_text_masked  replace every key word with "xxxx"   (reliance on key semantics)
           reading_order    e2e with words in plain reading order instead of gold entity order
Part E - value recall per field (seen vs unseen), from the evaluation
Part F - layout probes on the held-out test set (value recall: identity / shift_scale / mirror_x)
Part G - held-out test set: value recall by structural property of the gold value
         (value type, multi-line, table line item vs single field, page, vertical position, length)
Parts A-D need gold keys and links, so they run on FUNSD / WildReceipt documents; E-G cover VRDU.
For an in-domain unseen-template comparison in A-D, hold out some training templates too:
  python scripts/02_template_split.py ... --unseen-frac 0.15
"""
from __future__ import annotations

import copy
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from .metrics import doc_counts, fmt, summarize
from .predict import predict_doc
from .schema import (TAGS, edge_gap, entity_box, entity_text, median_word_height, read_jsonl,
                     relation_direction, count_lines)
from .viz import draw_entities_links, page_image

DIST_BINS = [(0, 1, "<1h"), (1, 3, "1-3h"), (3, 8, "3-8h"), (8, 1e9, ">8h")]


def value_type(text: str) -> str:
    t = text.strip()
    if re.fullmatch(r"[\d\s/\-.:]+", t) and re.search(r"\d", t):
        return "date-like" if re.search(r"\d{1,4}[/\-.]\d{1,2}[/\-.]\d{1,4}", t) else "numeric"
    if re.fullmatch(r"[^\d]+", t):
        return "alphabetic"
    return "mixed"


def link_properties(doc: dict) -> list[dict]:
    """One record per gold key->value link with its structural properties."""
    h = median_word_height(doc)
    fan = Counter(k for k, _ in doc["links"])
    out = []
    for k, v in doc["links"]:
        kb, vb = entity_box(doc, doc["entities"][k]), entity_box(doc, doc["entities"][v])
        gap = edge_gap(kb, vb) / h
        dist = next(lbl for lo, hi, lbl in DIST_BINS if lo <= gap < hi)
        vwords = [doc["words"][w]["box"] for w in doc["entities"][v]["word_ids"]]
        out.append({
            "key": k, "value": v,
            "direction": relation_direction(kb, vb),
            "distance": dist,
            "value_lines": "multi-line" if count_lines(vwords, 0.6 * h) > 1 else "single-line",
            "key_fanout": "one-to-many" if fan[k] > 1 else "one-to-one",
            "value_type": value_type(entity_text(doc, doc["entities"][v])),
            "source": doc["source"],
        })
    return out


def stratified(docs, preds, props_of) -> dict:
    """{property: {bucket: [n, correct]}}"""
    table = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for d in docs:
        gm = preds[d["doc_id"]]["gold_mode"]
        gold_keys = defaultdict(set)
        for k, v in d["links"]:
            gold_keys[v].add(k)
        for rec in props_of[d["doc_id"]]:
            ok = (gm.get(str(rec["value"])) or {}).get("key") in gold_keys[rec["value"]]
            for prop in ("direction", "distance", "value_lines", "key_fanout", "value_type", "source"):
                cell = table[prop][rec[prop]]
                cell[0] += 1
                cell[1] += int(ok)
    return table


def error_types(docs, preds) -> Counter:
    c = Counter()
    for d in docs:
        gm = preds[d["doc_id"]]["gold_mode"]
        gold_keys = defaultdict(set)
        for k, v in d["links"]:
            gold_keys[v].add(k)
        for v, ks in gold_keys.items():
            choice = (gm.get(str(v)) or {}).get("key")
            if choice in ks:
                c["correct"] += 1
            elif choice is None:
                c["predicted no key"] += 1
            else:
                vb = entity_box(d, d["entities"][v])
                d_pred = edge_gap(entity_box(d, d["entities"][choice]), vb)
                d_gold = min(edge_gap(entity_box(d, d["entities"][k]), vb) for k in ks)
                c["wrong key (nearer than gold)" if d_pred < d_gold else "wrong key (not nearer)"] += 1
        values_without_key = [e["id"] for e in d["entities"] if e["role"] == "value" and e["id"] not in gold_keys]
        for v in values_without_key:
            c["keyless value given a key" if (gm.get(str(v)) or {}).get("key") is not None
              else "keyless value correctly left alone"] += 1
    return c


def tag_confusion(docs, preds) -> dict:
    roles = ["KEY", "VALUE", "HEADER", "O"]
    m = {g: {p: 0 for p in roles} for g in roles}

    def role(t):
        return "O" if t == "O" else t.split("-", 1)[1]
    for d in docs:
        tags = ["O"] * len(d["words"])
        for e in d["entities"]:
            if e["role"] in ("key", "value", "header"):
                for w in e["word_ids"]:
                    tags[w] = "X-" + e["role"].upper()
        skip = set(d.get("ignore_words", []) or [])
        for w, pt in enumerate(preds[d["doc_id"]]["word_tags"]):
            if w not in skip:
                m[role(tags[w])][role(TAGS[pt])] += 1
    return m


def boundary_errors(docs, preds) -> dict:
    """Words with the right role but the wrong B/I prefix - they split or merge spans."""
    from .encoding import gold_word_tags
    n = wrong_prefix = 0
    for d in docs:
        for g, p in zip(gold_word_tags(d), preds[d["doc_id"]]["word_tags"]):
            if g >= 0 and TAGS[g] != "O":
                n += 1
                if TAGS[p] != "O" and TAGS[g][2:] == TAGS[p][2:] and TAGS[g][0] != TAGS[p][0]:
                    wrong_prefix += 1
    return {"tagged_words": n, "right_role_wrong_prefix": wrong_prefix}


# ----------------------------------------------------------------------- probes
def _move_entity(doc, ent, dx, dy):
    for w in ent["word_ids"]:
        b = doc["words"][w]["box"]
        doc["words"][w]["box"] = [min(max(b[0] + dx, 0), doc["width"]), min(max(b[1] + dy, 0), doc["height"]),
                                  min(max(b[2] + dx, 0), doc["width"]), min(max(b[3] + dy, 0), doc["height"])]


def make_probe(doc: dict, probe: str):
    """Edited deep copy of doc (image dropped: geometry edits would contradict the pixels)."""
    d = copy.deepcopy(doc)
    d["image_path"] = None
    W, H = d["width"], d["height"]
    h = median_word_height(d)
    if probe == "shift_scale":
        s, tx, ty = 0.8, 0.1 * W, 0.1 * H
        for w in d["words"]:
            b = w["box"]
            w["box"] = [b[0] * s + tx, b[1] * s + ty, b[2] * s + tx, b[3] * s + ty]
    elif probe == "mirror_x":
        for w in d["words"]:
            b = w["box"]
            w["box"] = [W - b[2], b[1], W - b[0], b[3]]
    elif probe in ("value_below", "value_right"):
        moved = set()
        for k, v in d["links"]:
            if v in moved:
                continue
            kb, vb = entity_box(d, d["entities"][k]), entity_box(d, d["entities"][v])
            if probe == "value_below":
                dx, dy = kb[0] - vb[0], kb[3] + 0.4 * h - vb[1]
            else:
                dx, dy = kb[2] + 1.0 * h - vb[0], kb[1] - vb[1]
            _move_entity(d, d["entities"][v], dx, dy)
            moved.add(v)
    elif probe == "key_text_masked":
        for e in d["entities"]:
            if e["role"] == "key":
                for w in e["word_ids"]:
                    d["words"][w]["text"] = "xxxx"
    elif probe not in ("identity", "reading_order"):
        raise ValueError(probe)
    return d


PROBES = ["identity", "shift_scale", "mirror_x", "value_below", "value_right", "key_text_masked", "reading_order"]


VALUE_ONLY_PROBES = ["identity", "shift_scale", "mirror_x"]   # usable without gold keys/links (VRDU)
LINE_ITEM_FIELDS = {"channel", "program_desc", "program_start_date", "program_end_date", "sub_amount"}


def value_strata(doc, gold, page_h):
    """Structural properties of one gold value on a value-only test page."""
    from .schema import count_lines
    props = {"value_type": value_type(gold["text"]),
             "value_lines": "multi-line" if "\n" in gold["text"].strip() else "single-line",
             "n_words": "1 word" if len(gold["text"].split()) == 1 else ("2-4 words" if len(gold["text"].split()) <= 4 else "5+ words"),
             "structure": "table line item" if gold.get("field") in LINE_ITEM_FIELDS else "single field",
             "page": "first page" if int(doc.get("page_index", 0)) == 0 else "later page"}
    ids = gold.get("word_ids") or []
    if ids:
        boxes = [doc["words"][i]["box"] for i in ids]
        h = median_word_height(doc)
        if count_lines(boxes, 0.6 * h) > 1:
            props["value_lines"] = "multi-line"
        y = (min(b[1] for b in boxes) + max(b[3] for b in boxes)) / 2
        props["vertical position"] = ["top third", "middle third", "bottom third"][min(2, int(3 * y / max(page_h, 1)))]
    return props


def test_set_strata(test_files, preds, fuzzy) -> dict:
    """Value recall on the held-out test pages, broken down by structural property of the gold value."""
    from .metrics import norm_text, predicted_value_candidates, value_match
    from .schema import iter_jsonl
    table = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for f in test_files:
        for d in iter_jsonl(f):
            p = preds.get(d["doc_id"])
            if p is None:
                continue
            cands = predicted_value_candidates(d, p)
            for g in d.get("gold_values", []):
                gt = norm_text(g["text"])
                if not gt:
                    continue
                found = any(value_match(t, gt, fuzzy) for t, _ in cands)
                for prop, bucket in value_strata(d, g, d["height"]).items():
                    cell = table[prop][bucket]
                    cell[0] += 1
                    cell[1] += int(found)
                cell = table["source"][d["source"]]
                cell[0] += 1
                cell[1] += int(found)
    return {p: {b: {"n": n, "recall": c / n if n else None} for b, (n, c) in sorted(t.items())} for p, t in table.items()}


def run_probes(model, encoder, docs, device, adt, fuzzy=0.8, probes=PROBES) -> dict:
    out = {}
    for probe in probes:
        counts = []
        for d in docs:
            pd = make_probe(d, probe)
            pred = predict_doc(model, encoder, pd, device, adt, use_gold_units=(probe != "reading_order"))
            counts.append(doc_counts(pd, pred, fuzzy))
        s = summarize(counts)
        out[probe] = {k: s[k] for k in ("value_recall", "value_acc", "link_f1", "entity_f1", "e2e_pair_f1")}
    return out


def sample_test_docs(files, n, seed=0):
    """Reservoir sample of n pages from the streamed test files."""
    import random
    from .schema import iter_jsonl
    rng, res, i = random.Random(seed), [], 0
    for f in files:
        for d in iter_jsonl(f):
            if len(res) < n:
                res.append(d)
            else:
                j = rng.randint(0, i)
                if j < n:
                    res[j] = d
            i += 1
    return res


# ----------------------------------------------------------------------- driver
def draw_link_errors(doc, pred, path):
    """Gold links red; wrong gold-mode choices magenta."""
    gm = pred["gold_mode"]
    gold_keys = defaultdict(set)
    for k, v in doc["links"]:
        gold_keys[v].add(k)
    wrong = [(info["key"], int(v)) for v, info in gm.items()
             if info["key"] is not None and info["key"] not in gold_keys.get(int(v), set())]
    im = page_image(doc)
    draw_entities_links(im, doc["words"], doc["entities"], doc["links"], wrong_links=wrong)
    im.save(path)
    return len(wrong)


def run_analysis(run_dir, docs_path=None, probe_docs: int = 60, log=print) -> dict:
    from .runtime import load_run
    run_dir = Path(run_dir)
    model, encoder, cfg, adt, _ = load_run(run_dir)
    device = next(model.parameters()).device
    docs = {d["doc_id"]: d for d in read_jsonl(docs_path or cfg["data"]["docs"])}
    pred_path = run_dir / "eval" / "predictions.jsonl"
    if not pred_path.exists():
        raise SystemExit("run scripts/05_evaluate.py first (needs eval/predictions.jsonl)")
    preds = {p["doc_id"]: p for p in read_jsonl(pred_path)}
    out_dir = run_dir / "analysis"
    (out_dir / "errors").mkdir(parents=True, exist_ok=True)

    result = {}
    sets = {}
    for split in ("seen_test", "unseen_test"):  # link-level analysis needs gold keys and links
        sdocs = [d for d in docs.values() if d.get("split") == split and d["doc_id"] in preds
                 and d.get("annotation", "full") == "full" and d["links"]]
        if sdocs:
            sets[split] = sdocs
    props = {d["doc_id"]: link_properties(d) for s in sets.values() for d in s}
    for split, sdocs in sets.items():
        result[split] = {
            "stratified": {p: {b: {"n": n, "recall": c / n if n else 0.0} for b, (n, c) in sorted(t.items())}
                           for p, t in stratified(sdocs, preds, props).items()},
            "errors": dict(error_types(sdocs, preds)),
            "tag_confusion": tag_confusion(sdocs, preds),
            "boundary": boundary_errors(sdocs, preds),
        }
        sample = sorted(sdocs, key=lambda d: d["doc_id"])[:probe_docs]
        log(f"[{split}] running {len(PROBES)} probes on {len(sample)} fully annotated documents ...")
        result[split]["probes"] = run_probes(model, encoder, sample, device, adt, cfg["eval"]["fuzzy_threshold"])

    # held-out test set (VRDU): per-field recall from the evaluation, and value-level layout probes
    mpath = run_dir / "eval" / "metrics.json"
    if mpath.exists():
        m = json.loads(mpath.read_text(encoding="utf-8"))
        result["fields"] = {sp: {"fields": v.get("fields", {}), "kv_fields": v.get("kv_fields", {})}
                            for sp, v in m.get("splits", {}).items() if sp in ("seen_test", "unseen_test")}
    test_files = [f for f in cfg["data"].get("test_docs", []) if Path(f).exists()]
    if test_files:
        log("[held-out test] value recall by structural property ...")
        result["test_strata"] = test_set_strata(test_files, preds, cfg["eval"]["fuzzy_threshold"])
    if test_files and probe_docs > 0:
        sample = sample_test_docs(test_files, probe_docs)
        log(f"[held-out test] running {len(VALUE_ONLY_PROBES)} value-level probes on {len(sample)} pages ...")
        result["test_probes"] = run_probes(model, encoder, sample, device, adt, cfg["eval"]["fuzzy_threshold"],
                                           probes=VALUE_ONLY_PROBES)

    # error images: unseen documents with the most wrong links
    if "unseen_test" in sets:
        ranked = sorted(sets["unseen_test"], key=lambda d: -sum(
            1 for v, info in preds[d["doc_id"]]["gold_mode"].items()
            if info["key"] is not None and info["key"] not in {k for k, vv in d["links"] if vv == int(v)}))
        for d in ranked[:8]:
            draw_link_errors(d, preds[d["doc_id"]], out_dir / "errors" / (d["doc_id"].replace("/", "__") + ".png"))

    with open(out_dir / "failure_analysis.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    report = format_report(result)
    (out_dir / "failure_report.md").write_text(report, encoding="utf-8")
    log(report)
    log(f"\nWrote {out_dir}/failure_report.md, failure_analysis.json, errors/")
    return result


def format_report(result: dict) -> str:
    S, U = result.get("seen_test"), result.get("unseen_test")
    L = ["# Failure analysis", ""]
    if S and U:
        L += ["## A. Link recall by structural property (gold entities)", "",
              "| property | bucket | n seen | recall seen | n unseen | recall unseen | drop |", "|---|---|---|---|---|---|---|"]
        for prop in U["stratified"]:
            buckets = sorted(set(S["stratified"].get(prop, {})) | set(U["stratified"][prop]))
            for b in buckets:
                s = S["stratified"].get(prop, {}).get(b, {"n": 0, "recall": 0.0})
                u = U["stratified"][prop].get(b, {"n": 0, "recall": 0.0})
                drop = f"{s['recall'] - u['recall']:+.3f}" if s["n"] and u["n"] else "n/a"
                L.append(f"| {prop} | {b} | {s['n']} | {s['recall']:.3f} | {u['n']} | {u['recall']:.3f} | {drop} |")
        L += ["", "Buckets with n < 20 are anecdotes, not evidence.", ""]
    for name, R in (("seen_test", S), ("unseen_test", U)):
        if not R:
            continue
        L += [f"## B. Error types - {name}", ""]
        tot = sum(v for k, v in R["errors"].items() if not k.startswith("keyless"))
        for k, v in sorted(R["errors"].items()):
            share = f" ({100 * v / tot:.1f}% of gold-linked values)" if tot and not k.startswith("keyless") else ""
            L.append(f"- {k}: {v}{share}")
        L += ["", f"## C. Tagging confusion - {name} (rows gold, columns predicted, words)", "",
              "| gold \\ pred | KEY | VALUE | HEADER | O |", "|---|---|---|---|---|"]
        for g, row in R["tag_confusion"].items():
            L.append(f"| {g} | " + " | ".join(str(row[p]) for p in ("KEY", "VALUE", "HEADER", "O")) + " |")
        bd = R["boundary"]
        L += ["", f"Span boundaries: {bd['right_role_wrong_prefix']} of {bd['tagged_words']} tagged words have the right role "
              "but the wrong B/I prefix (adjacent spans merged or one span split)."]
        L += ["", f"## D. Counterfactual probes - {name} (fully annotated documents)", "",
              "| probe | value recall | value acc | link F1 | entity F1 | e2e pair F1 |", "|---|---|---|---|---|---|"]
        base = R["probes"]["identity"]
        for p, m in R["probes"].items():
            d = (f" ({m['value_acc'] - base['value_acc']:+.3f})" if m["value_acc"] is not None and base["value_acc"] is not None else "")
            L.append(f"| {p} | {fmt(m['value_recall'])} | {fmt(m['value_acc'])}{d} | {fmt(m['link_f1'])} | "
                     f"{fmt(m['entity_f1'])} | {fmt(m['e2e_pair_f1'])} |")
        L.append("")
    if result.get("fields"):
        L += ["## E. Value recall per field (from the evaluation)", "",
              "| split | field | value recall | key-matched recall | n |", "|---|---|---|---|---|"]
        for sp, F in result["fields"].items():
            for f, v in F["fields"].items():
                kv = F["kv_fields"].get(f)
                L.append(f"| {sp} | {f} | {fmt(v['recall'])} | {fmt(kv['recall']) if kv else '-'} | {v['n']} |")
        L.append("")
    if result.get("test_strata"):
        L += ["## G. Held-out test set: value recall by structural property of the gold value", "",
              "| property | bucket | n | value recall |", "|---|---|---|---|"]
        for prop, buckets in result["test_strata"].items():
            for b, v in buckets.items():
                L.append(f"| {prop} | {b} | {v['n']} | {fmt(v['recall'])} |")
        L += ["", "Read across buckets of one property: a much lower recall for 'table line item', 'multi-line' or",
              "'later page' names the structural assumption that does not transfer. Buckets with n < 30 are anecdotal.", ""]
    if result.get("test_probes"):
        L += ["## F. Layout probes on the held-out test set (value recall)", "",
              "| probe | value recall | change |", "|---|---|---|"]
        b = result["test_probes"]["identity"]["value_recall"]
        for p, m in result["test_probes"].items():
            ch = f"{m['value_recall'] - b:+.3f}" if m["value_recall"] is not None and b is not None else "-"
            L.append(f"| {p} | {fmt(m['value_recall'])} | {ch} |")
        L.append("")
    L += ["How to read D: a large drop under value_below means the model assumes values sit to the right of",
          "keys; under mirror_x, that it assumes left-to-right layouts; under key_text_masked, that it relies on",
          "what keys say rather than where things are; shift_scale should be ~0 (relative geometry).",
          "Probes run without the page image (edited geometry would contradict the pixels), so always compare",
          "a probe with the identity row of the same table, not with the evaluation report.", ""]
    return "\n".join(L)

