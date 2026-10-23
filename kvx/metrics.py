"""Metrics. Every metric is built from per-document counts (tp, n_pred, n_gold), so it can be summed
per split, per source, per template, and bootstrapped over templates.

Defined on EVERY document that has gold values (FUNSD, SROIE, WildReceipt, VRDU)
  value_recall     share of gold values found as a predicted VALUE span. Text match (kvx.metrics.value_match):
                   equal alphanumerics, a short glued alphabetic prefix (currency), or - for mostly
                   non-numeric values only - similarity >= threshold. A multi-line gold value may also match the concatenation of
                   the predicted value spans linked to the same key.  <- the cross-dataset headline metric
  kv_field_recall  share of gold values found AND linked to a printed key whose text matches the field's
                   aliases (kvx/field_aliases.yaml). Only fields with aliases count.
Defined only where keys and links are annotated (annotation "full": FUNSD, WildReceipt)
  entity_f1        key/value spans, exact word sets
  link_f1          link head on GOLD spans (isolates linking)
  e2e_pair_f1      predicted spans + links; key and value word sets must both be exact
  e2e_fuzzy_f1     same, matched on normalised text
  value_acc        share of gold-linked values whose chosen key is a gold key
  value_precision  share of predicted value spans that match a gold value text
A metric with nothing to measure in a set of documents is None (shown as "-"), never 0.
"""
from __future__ import annotations

import random
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import yaml

from .schema import entity_text

HEADLINE = ["value_recall", "kv_field_recall", "entity_f1", "link_f1", "e2e_pair_f1"]
ALIASES_PATH = Path(__file__).with_name("field_aliases.yaml")


def f1(tp, npred, ngold):
    p = tp / npred if npred else 0.0
    r = tp / ngold if ngold else 0.0
    return (2 * p * r / (p + r) if p + r else 0.0), p, r


def norm_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).lower()
    s = re.sub(r"[^\w]+", " ", s, flags=re.UNICODE)
    return " ".join(s.split())


def similarity(a: str, b: str) -> float:
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    try:
        from rapidfuzz.distance import Levenshtein
        return 1.0 - Levenshtein.distance(a, b) / max(len(a), len(b))
    except ImportError:
        pass
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return 1.0 - prev[-1] / max(len(a), len(b))


def _alnum(s: str) -> str:
    return re.sub(r"[^0-9a-z]", "", unicodedata.normalize("NFKC", s).lower())


_DATE = re.compile(r"^\s*(\d{1,4})[/\-. ](\d{1,2})[/\-. ](\d{1,4})\s*$")


def _date_key(s: str):
    """Numeric date -> (a, b, year) with leading zeros and 2-digit years normalised, or None."""
    m = _DATE.match(s)
    if not m:
        return None
    a, b, c = (int(x) for x in m.groups())
    if a > 31:                       # yyyy-mm-dd
        a, b, c = b, c, a
    if c < 100:
        c += 2000 if c < 50 else 1900
    return a, b, c


def value_match(pred_norm: str, gold_norm: str, threshold: float) -> bool:
    """Same value? Exact on alphanumerics (ignores spacing/punctuation: "$1,000.00" = "1000.00"), or the
    prediction carries a short alphabetic prefix glued to it ("RM9.00" for "9.00"), or fuzzy similarity."""
    a, b = _alnum(pred_norm), _alnum(gold_norm)
    if b and a == b:
        return True
    da, db = _date_key(pred_norm), _date_key(gold_norm)
    if da is not None and db is not None:
        return da == db              # 6/5/2019 == 06/05/2019 == 06-05-19 (as VRDU's DateMatch does)
    if b and a.endswith(b) and 0 < len(a) - len(b) <= 3 and a[: len(a) - len(b)].isalpha():
        return True
    if b and sum(ch.isdigit() for ch in b) >= 0.5 * len(b):
        return False  # numbers, dates, amounts: one wrong digit is a wrong value - no fuzzy matching
    return similarity(pred_norm, gold_norm) >= threshold


_ALIASES = None


def load_aliases(path=None) -> dict:
    global _ALIASES
    if path is None and _ALIASES is not None:
        return _ALIASES
    p = Path(path) if path else ALIASES_PATH
    data = yaml.safe_load(p.read_text(encoding="utf-8")) if p.exists() else {}
    out = {f: [norm_text(a) for a in (al or []) if norm_text(a)] for f, al in (data or {}).items()}
    if path is None:
        _ALIASES = out
    return out


def key_matches(key_text: str, aliases: list[str]) -> bool:
    k = f" {norm_text(key_text)} "
    return any(f" {a} " in k for a in aliases)


def _sig(word_ids):
    return tuple(sorted(word_ids))


def _words_text(doc, ids):
    return " ".join(doc["words"][i]["text"] for i in ids)


def predicted_value_candidates(doc: dict, pred: dict) -> list[tuple[str, str | None]]:
    """(normalised value text, linked key text or None): every predicted value span, plus, per key, the
    concatenation of all its values in reading order (multi-line values split by OCR lines)."""
    ents = pred["pred_entities"]
    key_of = {v: k for k, v in pred["pred_links"]}
    cands, by_key = [], defaultdict(list)
    for i, e in enumerate(ents):
        if e["role"] != "value":
            continue
        k = key_of.get(i)
        ktxt = _words_text(doc, ents[k]["word_ids"]) if k is not None else None
        cands.append((norm_text(_words_text(doc, e["word_ids"])), ktxt))
        if k is not None:
            by_key[k].append(i)
    for k, vs in by_key.items():
        if len(vs) > 1:
            ids = [w for v in sorted(vs, key=lambda v: min(ents[v]["word_ids"])) for w in ents[v]["word_ids"]]
            cands.append((norm_text(_words_text(doc, ids)), _words_text(doc, ents[k]["word_ids"])))
    return cands


def gold_value_list(doc: dict) -> list[dict]:
    if doc.get("gold_values") is not None:
        return doc["gold_values"]
    return [{"field": None, "text": entity_text(doc, e)} for e in doc["entities"] if e["role"] == "value"]


def doc_counts(doc: dict, pred: dict, fuzzy_threshold: float = 0.8, aliases: dict | None = None) -> dict:
    aliases = load_aliases() if aliases is None else aliases
    c = {}
    # ---------------- value level (all annotated documents)
    cands = predicted_value_candidates(doc, pred)
    golds = gold_value_list(doc)
    tp_rec = tp_kv = n_kv = 0
    for g in golds:
        gt = norm_text(g["text"])
        if not gt:
            continue
        hits = [(t, k) for t, k in cands if value_match(t, gt, fuzzy_threshold)]
        found = bool(hits)
        tp_rec += found
        field = g.get("field")
        if field:
            c[f"field:{field}"] = tuple(a + b for a, b in zip(c.get(f"field:{field}", (0, 0, 0)), (int(found), 0, 1)))
        al = aliases.get(field) if field else None
        if al:
            n_kv += 1
            ok = any(k is not None and key_matches(k, al) for _, k in hits)
            tp_kv += ok
            c[f"kvfield:{field}"] = tuple(a + b for a, b in zip(c.get(f"kvfield:{field}", (0, 0, 0)), (int(ok), 0, 1)))
    c["value_rec"] = (tp_rec, 0, sum(1 for g in golds if norm_text(g["text"])))
    c["kv_field"] = (tp_kv, 0, n_kv)
    c["pred_values"] = (sum(1 for e in pred["pred_entities"] if e["role"] == "value"), 0, 1)

    if doc.get("annotation", "full") != "full":
        return c
    # ---------------- span / link level (keys and links annotated)
    gnorm = [norm_text(g["text"]) for g in golds]
    single = [t for t, _ in cands[: sum(1 for e in pred["pred_entities"] if e["role"] == "value")]]
    c["value_prec"] = (sum(1 for t in single if any(value_match(t, g, fuzzy_threshold) for g in gnorm)), len(single), 0)
    gold_by_role, pred_by_role = defaultdict(set), defaultdict(set)
    for e in doc["entities"]:
        gold_by_role[e["role"]].add(_sig(e["word_ids"]))
    for e in pred["pred_entities"]:
        pred_by_role[e["role"]].add(_sig(e["word_ids"]))
    for r in ("key", "value", "header"):
        g, p = gold_by_role[r], pred_by_role[r]
        c[f"entity_{r}"] = (len(g & p), len(p), len(g))
    gl = {tuple(x) for x in doc["links"]}
    pl = {tuple(x) for x in pred["gold_mode_links"]}
    c["link"] = (len(gl & pl), len(pl), len(gl))
    ents, pe = doc["entities"], pred["pred_entities"]
    gold_pairs = {(_sig(ents[k]["word_ids"]), _sig(ents[v]["word_ids"])) for k, v in doc["links"]}
    pred_pairs = {(_sig(pe[k]["word_ids"]), _sig(pe[v]["word_ids"])) for k, v in pred["pred_links"]}
    c["e2e_pair"] = (len(gold_pairs & pred_pairs), len(pred_pairs), len(gold_pairs))
    txt = lambda ids: norm_text(_words_text(doc, ids))  # noqa: E731
    g_txt = [(txt(k), txt(v)) for k, v in gold_pairs]
    used, tp = set(), 0
    for pk, pv in ((txt(k), txt(v)) for k, v in pred_pairs):
        best, bj = -1.0, None
        for j, (gk, gv) in enumerate(g_txt):
            if j not in used:
                s = min(similarity(pk, gk), similarity(pv, gv))
                if s >= fuzzy_threshold and s > best:
                    best, bj = s, j
        if bj is not None:
            used.add(bj)
            tp += 1
    c["e2e_fuzzy"] = (tp, len(pred_pairs), len(g_txt))
    gold_keys_of = defaultdict(set)
    for k, v in doc["links"]:
        gold_keys_of[v].add(k)
    ok = sum(1 for v, ks in gold_keys_of.items() if (pred["gold_mode"].get(str(v)) or {}).get("key") in ks)
    c["value_acc"] = (ok, 0, len(gold_keys_of))
    return c


# ------------------------------------------------------------------ aggregation
def add_counts(tot: dict, c: dict):
    for k, v in c.items():
        t = tot.setdefault(k, [0, 0, 0])
        t[0] += v[0]
        t[1] += v[1]
        t[2] += v[2]


def finalize(tot: dict, n_docs: int) -> dict:
    out = {"n_docs": n_docs}

    def rec(k):
        t = tot.get(k)
        return t[0] / t[2] if t and t[2] else None
    out["value_recall"] = rec("value_rec")
    out["kv_field_recall"] = rec("kv_field")
    out["value_acc"] = rec("value_acc")
    t = tot.get("value_prec")
    out["value_precision"] = t[0] / t[1] if t and t[1] else None
    pv = tot.get("pred_values")
    out["pred_values_per_doc"] = pv[0] / pv[2] if pv and pv[2] else None
    for k in ("link", "e2e_pair", "e2e_fuzzy", "entity_key", "entity_value", "entity_header"):
        t = tot.get(k)
        if t and (t[1] or t[2]):
            out[f"{k}_f1"], out[f"{k}_p"], out[f"{k}_r"] = f1(t[0], t[1], t[2])
        else:
            out[f"{k}_f1"] = out[f"{k}_p"] = out[f"{k}_r"] = None
    ek, ev = tot.get("entity_key"), tot.get("entity_value")
    if ek or ev:
        z = [0, 0, 0]
        ek, ev = ek or z, ev or z
        out["entity_f1"], out["entity_p"], out["entity_r"] = f1(ek[0] + ev[0], ek[1] + ev[1], ek[2] + ev[2])
    else:
        out["entity_f1"] = out["entity_p"] = out["entity_r"] = None
    present = [out[m] for m in ("value_recall", "entity_f1", "link_f1", "e2e_pair_f1") if out[m] is not None]
    out["mean"] = sum(present) / len(present) if present else 0.0
    out["n_gold_values"] = tot.get("value_rec", [0, 0, 0])[2]
    out["n_gold_links"] = tot.get("link", [0, 0, 0])[2]
    out["fields"] = {k.split(":", 1)[1]: {"recall": v[0] / v[2] if v[2] else None, "n": v[2]}
                     for k, v in sorted(tot.items()) if k.startswith("field:")}
    out["kv_fields"] = {k.split(":", 1)[1]: {"recall": v[0] / v[2] if v[2] else None, "n": v[2]}
                        for k, v in sorted(tot.items()) if k.startswith("kvfield:")}
    return out


def summarize(counts_list: list[dict]) -> dict:
    tot = {}
    for c in counts_list:
        add_counts(tot, c)
    return finalize(tot, len(counts_list))


def by_group(counts: dict, group_of: dict) -> dict:
    groups = defaultdict(list)
    for doc_id, c in counts.items():
        groups[group_of[doc_id]].append(c)
    return {g: summarize(cs) for g, cs in sorted(groups.items())}


def _template_totals(counts: dict, template_of: dict):
    g = {}
    n = defaultdict(int)
    for doc_id, c in counts.items():
        t = template_of[doc_id]
        add_counts(g.setdefault(t, {}), c)
        n[t] += 1
    return g, n, sorted(g)


def _resample(g, n, keys, rng):
    tot, nd = {}, 0
    for k in (rng.choice(keys) for _ in keys):
        add_counts(tot, g[k])
        nd += n[k]
    return finalize(tot, nd)


def _pct(vals):
    vals = sorted(v for v in vals if v is not None)
    if not vals:
        return None
    B = len(vals)
    return vals[int(0.025 * B)], vals[min(B - 1, int(0.975 * B))]


def bootstrap_ci(counts: dict, template_of: dict, metrics, B: int = 1000, seed: int = 0) -> dict:
    """95% CIs by resampling TEMPLATES (documents of one template are not independent)."""
    g, n, keys = _template_totals(counts, template_of)
    if len(keys) < 2 or B <= 0:
        return {m: None for m in metrics}
    rng = random.Random(seed)
    vals = defaultdict(list)
    for _ in range(B):
        s = _resample(g, n, keys, rng)
        for m in metrics:
            vals[m].append(s.get(m))
    return {m: _pct(vals[m]) for m in metrics}


def bootstrap_gap(counts_a: dict, counts_b: dict, template_of: dict, metrics, B: int = 1000, seed: int = 0) -> dict:
    """95% CI of metric(A) - metric(B); templates resampled independently within each set."""
    ga, na, ka = _template_totals(counts_a, template_of)
    gb, nb, kb = _template_totals(counts_b, template_of)
    if len(ka) < 2 or len(kb) < 2 or B <= 0:
        return {m: None for m in metrics}
    rng = random.Random(seed)
    vals = defaultdict(list)
    for _ in range(B):
        sa, sb = _resample(ga, na, ka, rng), _resample(gb, nb, kb, rng)
        for m in metrics:
            vals[m].append(sa[m] - sb[m] if sa.get(m) is not None and sb.get(m) is not None else None)
    return {m: _pct(vals[m]) for m in metrics}


def fmt(x, nd=3):
    return "-" if x is None else f"{x:.{nd}f}"
