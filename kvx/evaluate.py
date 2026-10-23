"""Evaluation: seen templates (held-out documents of training templates) vs the held-out test set
(VRDU, plus any training-source templates held out), with template-level confidence intervals.

Splits
  val          held-out TRAINING-source templates used for model selection (reported for completeness)
  seen_test    unseen documents of templates seen in training
  unseen_test  templates never seen in training: every VRDU page (data.test_docs) plus any
               training-source templates the split held out
The headline gap uses value_recall (defined on every dataset) and kv_field_recall.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from PIL import Image

from .metrics import HEADLINE, bootstrap_ci, bootstrap_gap, by_group, doc_counts, fmt, summarize
from .predict import predict_doc
from .runtime import load_run
from .schema import iter_jsonl, read_jsonl
from .viz import draw_gold, draw_prediction, page_image


def side_by_side(doc, pred):
    if doc.get("annotation", "full") == "full" and doc["entities"]:
        gold_sig = {(tuple(sorted(doc["entities"][k]["word_ids"])), tuple(sorted(doc["entities"][v]["word_ids"])))
                    for k, v in doc["links"]}
        a = draw_gold(doc)
        b = draw_prediction(doc, pred, gold_links=gold_sig)
    else:
        a = page_image(doc)
        b = draw_prediction(doc, pred)
    canvas = Image.new("RGB", (a.width + b.width + 20, max(a.height, b.height)), "white")
    canvas.paste(a, (0, 0))
    canvas.paste(b, (a.width + 20, 0))
    return canvas


def iter_split(docs, test_files, split, sources, max_test_docs=None):
    for d in docs:
        if d.get("split") == split and (not sources or d["source"] in sources):
            yield d
    if split == "unseen_test":
        n = 0
        for f in test_files:
            for d in iter_jsonl(f):
                if max_test_docs is not None and n >= max_test_docs:
                    return
                if not sources or d["source"] in sources or d["source"].split("-")[0] in sources:
                    n += 1
                    yield d


def run_evaluation(run_dir: str, docs_path: str | None = None, splits=None, test_docs=None,
                   max_test_docs: int | None = None, log=print) -> dict:
    run_dir = Path(run_dir)
    model, encoder, cfg, adt, ck = load_run(run_dir)
    device = next(model.parameters()).device
    docs = read_jsonl(docs_path or cfg["data"]["docs"])
    test_files = list(test_docs if test_docs is not None else cfg["data"].get("test_docs", []))
    for f in test_files:
        if not Path(f).exists():
            raise FileNotFoundError(f"test docs file not found: {f}")
    src = cfg["data"]["eval_sources"]
    splits = splits or cfg["eval"]["splits"]
    fuzzy = cfg["eval"]["fuzzy_threshold"]
    out_dir = run_dir / "eval"
    (out_dir / "qualitative").mkdir(parents=True, exist_ok=True)
    B = cfg["eval"]["bootstrap"]
    tmpl, source = {}, {}
    results = {"run": str(run_dir), "checkpoint_epoch": ck.get("epoch"), "test_files": test_files, "splits": {}}
    all_counts = {}
    n_q = {"seen_test": 0, "unseen_test": 0}
    n_q_src = defaultdict(int)  # qualitative examples per (split, source)
    q_templates = set()
    with open(out_dir / "predictions.jsonl", "w", encoding="utf-8") as pf:
        for split in splits:
            counts, lost, multi = {}, 0, 0
            for i, d in enumerate(iter_split(docs, test_files, split, src, max_test_docs)):
                p = predict_doc(model, encoder, d, device, adt)
                counts[d["doc_id"]] = doc_counts(d, p, fuzzy)
                tmpl[d["doc_id"]], source[d["doc_id"]] = d["template_id"], d["source"]
                lost += p["lost_words"]
                multi += p["n_chunks"] > 1
                pf.write(json.dumps({**p, "split": split}) + "\n")
                per_src = max(1, cfg["eval"]["n_qualitative"] // 4)
                if split in n_q and d["template_id"] not in q_templates and n_q_src[(split, d["source"])] < per_src:
                    side_by_side(d, p).save(out_dir / "qualitative" / f"{split}__{d['doc_id'].replace('/', '__').replace('#', '_')}.png")
                    q_templates.add(d["template_id"])
                    n_q_src[(split, d["source"])] += 1
                if (i + 1) % 200 == 0:
                    log(f"  [{split}] {i + 1} documents evaluated ...")
            if not counts:
                log(f"[{split}] no documents - skipped")
                continue
            all_counts[split] = counts
            summ = summarize(list(counts.values()))
            summ["n_templates"] = len({tmpl[d] for d in counts})
            summ["ci95"] = bootstrap_ci(counts, tmpl, HEADLINE, B)
            summ["per_source"] = by_group(counts, source)
            summ["lost_words"] = lost
            summ["multi_chunk_docs"] = multi
            results["splits"][split] = summ
            per_t = by_group(counts, tmpl)
            with open(out_dir / f"per_template_{split}.csv", "w", encoding="utf-8") as f:
                f.write("template_id,n_docs," + ",".join(HEADLINE) + "\n")
                for t, m in per_t.items():
                    f.write(f"{t},{m['n_docs']}," + ",".join(fmt(m[h], 4) for h in HEADLINE) + "\n")

    if "seen_test" in all_counts and "unseen_test" in all_counts:
        S, U = results["splits"]["seen_test"], results["splits"]["unseen_test"]
        gap_ci = bootstrap_gap(all_counts["seen_test"], all_counts["unseen_test"], tmpl, HEADLINE, B)
        results["gap"] = {}
        for m in HEADLINE:
            if S[m] is None or U[m] is None:
                continue
            d = S[m] - U[m]
            results["gap"][m] = {"seen": S[m], "unseen": U[m], "gap": d,
                                 "relative_drop": d / S[m] if S[m] else None, "gap_ci95": gap_ci[m]}
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    txt = format_report(results)
    (out_dir / "report.txt").write_text(txt + "\n", encoding="utf-8")
    log(txt)
    log(f"\nWrote {out_dir}/metrics.json, report.txt, per_template_*.csv, predictions.jsonl, qualitative/")
    return results


def format_report(results: dict) -> str:
    L = [f"Run: {results['run']}   (best epoch {results['checkpoint_epoch']})", ""]
    L.append(f"{'split':12s} {'docs':>6s} {'tmpl':>5s} " + " ".join(f"{m:>16s}" for m in HEADLINE))
    for split, s in results["splits"].items():
        L.append(f"{split:12s} {s['n_docs']:6d} {s['n_templates']:5d} " + " ".join(f"{fmt(s[m]):>16s}" for m in HEADLINE))
    L.append("  ('-' = not measurable on these documents: VRDU and SROIE annotate values only, no keys or links)")
    if results.get("gap"):
        L += ["", "Seen - unseen gap (95% CI over templates):"]
        for m, g in results["gap"].items():
            ci = g["gap_ci95"]
            ci_s = f"[{ci[0]:+.3f}, {ci[1]:+.3f}]" if ci else "(too few templates for a CI)"
            rel = f"{100 * g['relative_drop']:.1f}%" if g["relative_drop"] is not None else "n/a"
            L.append(f"  {m:16s} seen {g['seen']:.3f}  unseen {g['unseen']:.3f}  gap {g['gap']:+.3f} {ci_s}  relative drop {rel}")
    L += ["", "Per source:"]
    for split, s in results["splits"].items():
        for so, m in s["per_source"].items():
            L.append(f"  {split:12s} {so:15s} docs {m['n_docs']:5d}  value-rec {fmt(m['value_recall'])}  "
                     f"kv-field {fmt(m['kv_field_recall'])}  entity {fmt(m['entity_f1'])}  link {fmt(m['link_f1'])}  "
                     f"e2e {fmt(m['e2e_pair_f1'])}  values/doc {fmt(m['pred_values_per_doc'], 1)}")
    for split, s in results["splits"].items():
        if s.get("fields"):
            L += ["", f"Per field ({split}): value recall | key-matched recall | n"]
            for f, v in s["fields"].items():
                kv = s.get("kv_fields", {}).get(f)
                L.append(f"  {f:24s} {fmt(v['recall'])} | {fmt(kv['recall']) if kv else '   - '} | {v['n']}")
    return "\n".join(L)
