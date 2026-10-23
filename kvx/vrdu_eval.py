"""VRDU benchmark-style field extraction scoring (Option E) from SAVED predictions - no model run.

VRDU labels field values, not printed keys, so entity / link / pair F1 cannot be computed on it. What CAN be
computed is the benchmark's own metric: per document, the extracted value of each field is compared with the gold
value using VRDU's type-specific match functions (meta.json), giving precision, recall and F1 (micro, macro, per
field, unrepeated fields). The matching and scoring below are a faithful port of google-research/vrdu
(match_utils.py, benchmark_utils.py; Apache License 2.0).

Our model is schema-free: it outputs (printed key, value) pairs. To score it on VRDU's schema, each predicted pair
is mapped to a field by its KEY TEXT (kvx/field_aliases.yaml):
  * fields of the document's corpus whose alias occurs in the key as whole words; the longest alias wins
  * ties: keep fields whose type fits the value (date / price / number); then a key that owns several values
    prefers table (line_item) fields, a key with one value prefers single (unrepeated) fields
  * a value holding two dates for a from/to pair ("10/01/20 - 10/07/20") is split: first -> from, second -> to
  * values with no key, or whose key names no field, are not extracted (the schema-free model cannot name them)
Unrepeated fields: the most probable link is kept (VRDU keeps the first extraction per field).
Line items: extractions are ordered top-to-bottom, left-to-right and grouped by VRDU's own rule.
The alias list is part of this protocol - report it with the numbers.
"""
from __future__ import annotations

import collections
import datetime
import gzip
import json
import random
import re
from pathlib import Path

from .loaders_extra import VRDU_CORPORA, slug
from .metrics import load_aliases, norm_text
from .schema import iter_jsonl, reading_order_key, union_box, median_word_height


# ============================================================ match functions (port of vrdu/match_utils.py)
def _ws(s: str) -> str:
    return " ".join(s.strip().split())


def _alnum_eq(a, b):
    return re.sub(r"[^0-9a-zA-Z]", "", a) == re.sub(r"[^0-9a-zA-Z]", "", b)


def _nows_eq(a, b):
    return re.sub(r"\s", "", a) == re.sub(r"\s", "", b)


def _value_eq(a, b, diff=0.01):
    try:
        return abs(float(re.sub(r"[^0-9.]", "", a)) - float(re.sub(r"[^0-9.]", "", b))) <= diff
    except ValueError:
        return False


def _num_eq(a, b):
    return re.sub(r"[^0-9]", "", a) == re.sub(r"[^0-9]", "", b)


def _edit_distance(a: str, b: str) -> int:
    try:
        from rapidfuzz.distance import Levenshtein
        return Levenshtein.distance(a, b)
    except ImportError:
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            cur = [i]
            for j, cb in enumerate(b, 1):
                cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
            prev = cur
        return prev[-1]


_DATE_PATTERNS = ["%m/%d/%y", "%m/%d/%Y", "%m/%d", "%b%d/%y", "%m-%d-%Y", "%m-%d-%y", "%B%d,%Y", "%Y/%m/%d"]


def decode_date(s: str):
    p = re.sub(r"[^0-9a-zA-Z/\-,]", "", s)
    for pat in _DATE_PATTERNS:
        try:
            d = datetime.datetime.strptime(p, pat).date()
            return d.year, d.month, d.day
        except ValueError:
            continue
    return None


def _date_eq(a, b):
    da, db = decode_date(a), decode_date(b)
    if not da or not db:
        return _alnum_eq(a, b)
    return da == db or (da[0] == db[0] and da[1] == db[2] and da[2] == db[1])   # day/month swap allowed


MATCHERS = {
    "DateMatch": _date_eq,
    "PriceMatch": _value_eq,
    "AddressMatch": lambda a, b: _edit_distance(a, b) <= 3,
    "NumericalStringMatch": _num_eq,
    "GeneralStringMatch": _alnum_eq,
    "NameMatch": _nows_eq,
    "DefaultMatch": lambda a, b: a == b,
}


def match(func: str, extracted: str, candidates: list[str]) -> bool:
    """True if the extracted text matches ANY gold candidate (VRDU lists every appearance of a value)."""
    f = MATCHERS.get(func, MATCHERS["DefaultMatch"])
    e = _ws(extracted)
    return any(f(e, _ws(c)) for c in candidates)


# ============================================================ gold (VRDU's dataset.jsonl, read directly)
def load_meta(main: Path) -> dict:
    meta = json.loads((main / "meta.json").read_text(encoding="utf-8"))
    funcs = meta["entity_name_to_match_func"]
    pattern = meta.get("entity_appearance_pattern", {n: "unrepeated" for n in funcs})
    groups = collections.defaultdict(list)
    for n, p in pattern.items():
        groups[p].append(n)
    groups.setdefault("unrepeated", [])
    groups.setdefault("repeated", [])
    return {"name": meta.get("dataset_name"), "funcs": funcs, "pattern": pattern, "groups": dict(groups)}


def _open(main: Path):
    for name in ("dataset.jsonl", "dataset.jsonl.gz"):
        p = main / name
        if p.exists():
            return gzip.open(p, "rt", encoding="utf-8") if name.endswith(".gz") else open(p, encoding="utf-8")
    raise FileNotFoundError(f"no dataset.jsonl(.gz) in {main}")


def load_gold(main: Path) -> dict:
    """{filename: [(entity_name, [candidate, ...]) | ((sub_names...), [(sub_text, ...)]), ...]} plus the page of
    each item. Simple item: candidates = list of (text, page). Nested item: one tuple of (text, page) per sub-entity."""
    gold = {}
    with _open(main) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            items = []
            for ent in rec.get("annotations") or []:
                name, ents = ent[0], ent[1]
                if isinstance(name, list):          # nested line item: one candidate, sub-entities in name order
                    subs = tuple((e[0], int(e[1][0])) for e in ents[0])
                    items.append((tuple(name), [subs]))
                else:
                    items.append((name, [(e[0], int(e[1][0])) for e in ents]))
            gold[rec["filename"]] = items
    return gold


# ============================================================ extractions from saved predictions
def _is_date(t):
    return decode_date(_ws(t)) is not None or bool(re.search(r"\d{1,4}[/\-.]\d{1,2}([/\-.]\d{1,4})?", t))


def _fits(func: str, text: str) -> bool:
    if func == "DateMatch":
        return _is_date(text) or bool(re.search(r"(?i)\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)", text))
    if func in ("PriceMatch", "NumericalStringMatch"):
        return bool(re.search(r"\d", text))
    return True


_FROM_TO = [("flight_from", "flight_to"), ("program_start_date", "program_end_date")]
_DATE_RE = re.compile(r"\d{1,4}[/\-.]\d{1,2}(?:[/\-.]\d{1,4})?")


def type_fit(func: str, text: str) -> int:
    """How well a value's surface form fits a field's data type: 2 = typical, 1 = possible, 0 = implausible."""
    t = text.strip()
    date_like = _fits("DateMatch", t)
    has_digit = any(ch.isdigit() for ch in t)
    has_word = bool(re.search(r"[A-Za-z]{2,}", t))
    if func == "DateMatch":
        return 2 if date_like else 0
    if func == "PriceMatch":
        return 2 if re.search(r"\$|\d[.,]\d{2}\b", t) else (1 if has_digit else 0)
    if func == "NumericalStringMatch":
        return (1 if date_like else 2) if has_digit else 0
    if date_like and not has_word:          # names / strings / addresses: a bare date or number does not fit
        return 0
    return 1 if has_word else 0


def fields_for_key(key_text: str, value_text: str, n_values_of_key: int, meta: dict, aliases: dict) -> list[str]:
    """Field(s) a predicted (key, value) pair is evidence for; [] if the key names no field of this corpus.
    Among fields whose alias occurs in the key (whole words), rank by (type fit of the value, alias length):
    "Date of signature" + "06/05/2019" -> file_date, not signer_name, although "signature" is the longer alias."""
    k = f" {norm_text(key_text)} "
    alias_len = {}
    for field in meta["funcs"]:
        for a in aliases.get(field, []) or []:
            if f" {a} " in k:
                alias_len[field] = max(alias_len.get(field, 0), len(a))
    if not alias_len:
        return []
    rank = {f: (type_fit(meta["funcs"][f], value_text), n) for f, n in alias_len.items()}
    best = max(rank.values())
    cands = [f for f in meta["funcs"] if rank.get(f) == best]
    if len(cands) > 1:
        want = "line_item" if n_values_of_key > 1 else "unrepeated"
        pref = [c for c in cands if (meta["pattern"].get(c) == want) == (want == "line_item")]
        cands = pref or cands
    if len(cands) > 1:
        for a, b in _FROM_TO:                      # a from/to pair named by one key ("Flight dates")
            if a in cands and b in cands:
                return [a, b]
    return cands[:1]


def page_extractions(doc: dict, pred: dict, meta: dict, aliases: dict, stats: collections.Counter) -> list[dict]:
    """-> [{"field", "text", "p", "pos"}] for one predicted page."""
    words, ents = doc["words"], pred["pred_entities"]
    probs = pred.get("pred_link_p") or [None] * len(pred["pred_links"])
    n_vals = collections.Counter(k for k, _ in pred["pred_links"])
    tol = median_word_height(doc)
    out = []
    linked = set()
    for (k, v), p in zip(pred["pred_links"], probs):
        linked.add(v)
        ktxt = " ".join(words[i]["text"] for i in ents[k]["word_ids"])
        ids = sorted(ents[v]["word_ids"], key=lambda i: reading_order_key(words[i]["box"], tol))
        vtxt = " ".join(words[i]["text"] for i in ids)
        pos = (int(doc.get("page_index", 0)),) + reading_order_key(union_box(words[i]["box"] for i in ids), tol)
        fields = fields_for_key(ktxt, vtxt, n_vals[k], meta, aliases)
        if not fields:
            stats["pairs_key_names_no_field"] += 1
            continue
        if len(fields) == 2:                       # from/to: split two dates, else give the value to "from"
            dates = _DATE_RE.findall(vtxt)
            if len(dates) >= 2:
                out += [{"field": fields[0], "text": dates[0], "p": p, "pos": pos},
                        {"field": fields[1], "text": dates[1], "p": p, "pos": pos}]
                stats["pairs_split_from_to"] += 1
                continue
            fields = fields[:1]
        out.append({"field": fields[0], "text": vtxt, "p": p, "pos": pos})
        stats["pairs_mapped_to_field"] += 1
    stats["values_without_key"] += sum(1 for i, e in enumerate(ents) if e["role"] == "value" and i not in linked)
    return out


def doc_extractions(exts: list[dict], meta: dict) -> list[tuple]:
    """VRDU extraction list for one document: unrepeated fields ordered by link probability (VRDU keeps the first),
    line-item fields in reading order and grouped into nested entities by VRDU's rule."""
    unrep = set(meta["groups"].get("unrepeated", []))
    simple = [e for e in exts if meta["pattern"].get(e["field"], "unrepeated") in ("unrepeated", "repeated")]
    simple.sort(key=lambda e: (-(e["p"] if e["p"] is not None else 0.0), e["pos"]))
    items, seen = [], set()
    for e in simple:                                           # remove_redundant_unrepeated_entities
        if e["field"] in unrep:
            if e["field"] in seen:
                continue
            seen.add(e["field"])
        items.append((e["field"], e["text"]))
    for pat, names in meta["groups"].items():                  # group_repeated_entities_into_nested_entities
        if pat in ("unrepeated", "repeated"):
            continue
        rep = sorted((e for e in exts if e["field"] in names), key=lambda e: e["pos"])
        groups, cur = [], set()
        for e in rep:
            if not groups or e["field"] in cur:
                groups.append([])
                cur = set()
            groups[-1].append(e)
            cur.add(e["field"])
        for g in groups:
            items.append((tuple(e["field"] for e in g), tuple(e["text"] for e in g)))
    return items


# ============================================================ scoring (port of vrdu/benchmark_utils.py)
def _nested_name(names: tuple, nested: dict) -> str:
    s = set(names)
    for nn, subs in nested.items():
        if s.issubset(set(subs)):
            return nn
    raise ValueError(f"unknown nested entity {names}")


def _canonical(name, value, order: list):
    """Nested items: put sub-fields in meta.json order on BOTH sides, so a predicted row and a gold row with the
    same set of sub-fields share a key whatever order they were written in (VRDU compares sub-entities by name)."""
    if not isinstance(name, tuple):
        return name, value
    idx = sorted(range(len(name)), key=lambda i: order.index(name[i]) if name[i] in order else len(order))
    return tuple(name[i] for i in idx), tuple(value[i] for i in idx)


def match_doc(gold_items: list, ex_items: list, meta: dict, target=None) -> tuple[int, int, int]:
    """-> (correct, n_gold, n_extracted) for one document; same counting as VRDU's get_matching_result_per_doc:
    correct = matched pairs, n_gold = matched + unmatched gold, n_extracted = matched + unmatched extractions."""
    nested = {p: n for p, n in meta["groups"].items() if p not in ("unrepeated", "repeated")}
    order = list(meta["funcs"])

    def keep(name):
        if target is None:
            return True
        return (_nested_name(name, nested) if isinstance(name, tuple) else name) in target
    g = collections.defaultdict(list)
    x = collections.defaultdict(list)
    for name, cands in gold_items:
        if keep(name):
            if isinstance(name, tuple):
                name, sub = _canonical(name, cands[0], order)
                cands = [sub]
            g[name].append(cands)
    for name, ent in ex_items:
        if keep(name):
            name, ent = _canonical(name, ent, order)
            x[name].append(ent)
    matched = unmatched_gold = unmatched_ex = 0
    for name in set(g) - set(x):
        unmatched_gold += len(g[name])
    for name in set(x) - set(g):
        unmatched_ex += len(x[name])
    for name in set(g) & set(x):
        hit = []
        for cands in g[name]:
            found = None
            for ent in x[name]:
                if isinstance(name, tuple):
                    gd = dict(zip(name, cands[0]))
                    xd = dict(zip(name, ent))
                    ok = all(match(meta["funcs"].get(s, "DefaultMatch"), xd[s], [gd[s][0]]) for s in gd)
                else:
                    ok = match(meta["funcs"].get(name, "DefaultMatch"), ent, [c[0] for c in cands])
                if ok:
                    found = ent
                    break
            if found is None:
                unmatched_gold += 1
            else:
                matched += 1
                hit.append(found)
        unmatched_ex += sum(1 for ent in x[name] if ent not in hit)
    return matched, matched + unmatched_gold, matched + unmatched_ex


def prf(c, g, x) -> dict:
    """VRDU's convention: no extractions -> 0; no gold -> precision 0, recall 1 (flagged by n_gold = 0)."""
    if x == 0:
        p = r = f = 0.0
    elif g == 0:
        p, r, f = 0.0, 1.0, 0.0
    elif c == 0:
        p = r = f = 0.0
    else:
        p, r = c / x, c / g
        f = 2 * p * r / (p + r)
    return {"precision": p, "recall": r, "f1": f, "correct": c, "n_gold": g, "n_extracted": x}


def score(gold: dict, ext: dict, meta: dict, filenames, template_of: dict | None = None, B: int = 1000,
          seed: int = 0) -> dict:
    """Micro, macro, per-field and unrepeated precision / recall / F1 over `filenames` (VRDU's evaluate())."""
    fields = list(meta["groups"].get("unrepeated", [])) + list(meta["groups"].get("repeated", [])) + \
        [p for p in meta["groups"] if p not in ("unrepeated", "repeated")]
    per_doc = {}
    for fn in filenames:
        per_doc[fn] = {"all": match_doc(gold[fn], ext.get(fn, []), meta)}
        for name in fields:
            per_doc[fn][name] = match_doc(gold[fn], ext.get(fn, []), meta, {name})
        per_doc[fn]["unrepeated"] = match_doc(gold[fn], ext.get(fn, []), meta, set(meta["groups"].get("unrepeated", [])))

    def total(key, fns):
        c = g = x = 0
        for fn in fns:
            a, b, d = per_doc[fn][key]
            c, g, x = c + a, g + b, x + d
        return c, g, x
    out = {"n_docs": len(per_doc), "micro": prf(*total("all", per_doc)), "fields": {}}
    for name in fields:
        out["fields"][name] = prf(*total(name, per_doc))
    vals = [v for v in out["fields"].values() if v["n_gold"] > 0]   # fields absent from these docs are skipped
    out["macro"] = {m: (sum(v[m] for v in vals) / len(vals) if vals else None) for m in ("precision", "recall", "f1")}
    out["unrepeated"] = prf(*total("unrepeated", per_doc))
    if template_of and B > 0:                                       # 95% CI of micro F1, resampling templates
        by_t = collections.defaultdict(list)
        for fn in per_doc:
            by_t[template_of.get(fn, fn)].append(fn)
        keys = sorted(by_t)
        if len(keys) >= 2:
            rng, f1s = random.Random(seed), []
            for _ in range(B):
                fns = [fn for t in (rng.choice(keys) for _ in keys) for fn in by_t[t]]
                f1s.append(prf(*total("all", fns))["f1"])
            f1s.sort()
            out["micro"]["f1_ci95"] = (f1s[int(0.025 * B)], f1s[min(B - 1, int(0.975 * B))])
        out["n_templates"] = len(keys)
    return out


# ============================================================ driver
def run(run_dir, vrdu_root, test_docs=None, split_files=(), corpora=("registration-form", "ad-buy-form"),
        aliases_path=None, B: int = 1000, log=print) -> dict:
    run_dir, vrdu_root = Path(run_dir), Path(vrdu_root)
    pred_path = run_dir / "eval" / "predictions.jsonl"
    if not pred_path.exists():
        raise SystemExit(f"{pred_path} not found - run scripts/05_evaluate.py first")
    if test_docs is None:
        import yaml
        cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
        test_docs = cfg["data"].get("test_docs") or ["cache/vrdu.jsonl"]
    aliases = load_aliases(aliases_path)

    log("reading saved predictions ...")
    preds = {}
    for p in iter_jsonl(pred_path):
        if str(p["doc_id"]).startswith("vrdu/"):
            preds[p["doc_id"]] = {k: p.get(k) for k in ("pred_entities", "pred_links", "pred_link_p")}
    log(f"  {len(preds)} VRDU page predictions")

    split_names = set()
    for sf in split_files:
        split_names |= set(json.loads(Path(sf).read_text(encoding="utf-8"))["test"])

    results = {"run": str(run_dir), "test_docs": [str(t) for t in test_docs], "split_files": [str(s) for s in split_files],
               "aliases": str(aliases_path or "kvx/field_aliases.yaml"), "corpora": {}}
    for corpus in corpora:
        main = vrdu_root / corpus / "main"
        if not main.is_dir():
            log(f"[{corpus}] missing {main} - skipped")
            continue
        short = VRDU_CORPORA.get(corpus, slug(corpus))
        meta = load_meta(main)
        log(f"[{corpus}] reading gold annotations ...")
        gold = load_gold(main)
        stem2fn = {slug(Path(fn).stem, 100): fn for fn in gold}
        stats = collections.Counter()
        page_exts = collections.defaultdict(list)
        predicted_pages = collections.defaultdict(set)
        template_of = {}
        for tf in test_docs:
            for d in iter_jsonl(tf):
                if d.get("source") != f"vrdu-{short}":
                    continue
                p = preds.get(d["doc_id"])
                stem = d["doc_id"].split("/", 2)[2].split("#p")[0]
                fn = stem2fn.get(stem)
                if fn is None:
                    stats["pages_not_in_dataset_jsonl"] += 1
                    continue
                template_of[fn] = d.get("template_id")
                if p is None:
                    stats["pages_without_saved_prediction"] += 1
                    continue
                predicted_pages[fn].add(int(d.get("page_index", 0)))
                page_exts[fn] += page_extractions(d, p, meta, aliases, stats)
        ext = {fn: doc_extractions(e, meta) for fn, e in page_exts.items()}

        def item_pages(cands):
            return {c[1] for c in cands} if not isinstance(cands[0][0], tuple) else {s[1] for s in cands[0]}

        def covered_gold(fn):                      # gold items with at least one appearance on a predicted page
            return [(n, c) for n, c in gold[fn] if item_pages(c) & predicted_pages[fn]]
        all_fns = sorted(fn for fn in gold if gold[fn] and (not split_names or fn in split_names))
        cov_fns = [fn for fn in all_fns if predicted_pages.get(fn)]
        n_items = sum(len(gold[fn]) for fn in all_fns)
        n_cov = sum(len(covered_gold(fn)) for fn in cov_fns)
        cov_gold = {fn: covered_gold(fn) for fn in cov_fns}
        r = {"dataset": meta["name"], "match_functions": meta["funcs"], "appearance": meta["pattern"],
             "coverage": {"docs": len(all_fns), "docs_with_predicted_pages": len(cov_fns),
                          "gold_items": n_items, "gold_items_on_predicted_pages": n_cov},
             "all_pages": score(gold, ext, meta, all_fns, template_of, B),
             "predicted_pages": score(cov_gold, ext, meta, cov_fns, template_of, B),
             "mapping": dict(stats)}
        results["corpora"][corpus] = r
    if not results["corpora"]:
        raise SystemExit(f"no VRDU corpus found under {vrdu_root.resolve()} - expected "
                         f"{vrdu_root / 'registration-form' / 'main' / 'meta.json'} (is --vrdu right? is $D set?)")
    out = run_dir / "eval"
    (out / "vrdu_benchmark.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    txt = format_report(results)
    (out / "vrdu_benchmark.txt").write_text(txt + "\n", encoding="utf-8")
    log(txt)
    log(f"\nWrote {out / 'vrdu_benchmark.json'} and {out / 'vrdu_benchmark.txt'}")
    return results


def _f(x):
    return "-" if x is None else f"{x:.3f}"


def format_report(res: dict) -> str:
    L = ["VRDU benchmark-style field extraction (Option E) - scored from saved predictions",
         f"Run: {res['run']}   aliases: {res['aliases']}"
         + (f"   split files: {', '.join(res['split_files'])}" if res["split_files"] else "   documents: all annotated"), ""]
    for corpus, r in res["corpora"].items():
        cv = r["coverage"]
        L.append(f"== {corpus} ({r['dataset']}) ==")
        L.append(f"  documents {cv['docs']} | with predicted pages {cv['docs_with_predicted_pages']} | gold items "
                 f"{cv['gold_items']} | on predicted pages {cv['gold_items_on_predicted_pages']}")
        for view in ("all_pages", "predicted_pages"):
            s = r[view]
            ci = s["micro"].get("f1_ci95")
            L.append(f"  [{view:15s}] micro P {_f(s['micro']['precision'])} R {_f(s['micro']['recall'])} "
                     f"F1 {_f(s['micro']['f1'])}" + (f" [{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "")
                     + f" | macro F1 {_f(s['macro']['f1'])} | unrepeated F1 {_f(s['unrepeated']['f1'])}")
        L.append(f"  {'field':22s} {'match':22s} {'P':>6s} {'R':>6s} {'F1':>6s} {'gold':>6s} {'extr':>6s}")
        for name, v in r["all_pages"]["fields"].items():
            func = r["match_functions"].get(name, "nested")
            L.append(f"  {name:22s} {func:22s} {_f(v['precision']):>6s} {_f(v['recall']):>6s} {_f(v['f1']):>6s} "
                     f"{v['n_gold']:6d} {v['n_extracted']:6d}")
        m = r["mapping"]
        L.append("  mapping: " + ", ".join(f"{k}={v}" for k, v in sorted(m.items())))
        L.append("")
    L += ["all_pages = every annotated document (gold on pages the model never saw counts as missed);",
          "predicted_pages = only gold appearing on pages that were predicted (re-run 01b with --all-pages + step 5",
          "to cover every page). Matching and scoring follow google-research/vrdu; fields are named from the printed",
          "key via the alias list, so values without a printed key cannot be extracted by this schema-free model."]
    return "\n".join(L)
