"""Template induction and the zero-same-template split.

FUNSD and XFUND do not label templates, so we induce them:
  * layout fingerprint  - fraction of each GxG page cell covered by word boxes
  * key-text fingerprint - TF-IDF of KEY and HEADER texts (values change per instance, so excluded)
  distance = alpha * (1 - cos_layout) + (1 - alpha) * (1 - cos_text)
and cluster with average-linkage agglomerative clustering cut at a distance threshold.
The threshold must be audited by eye (montages are written for every multi-document
cluster) - that audit is part of the deliverable, not a formality.

Custom forms already carry their template id (their folder name).

Split, per source:
  unseen_test : whole templates, never seen in training                 -> F1_unseen
  val         : whole templates, never trained on; used ONLY for model selection
  train       : remaining templates
  seen_test   : documents held out from train templates that have >= 2 documents
                (unseen document, seen template)                         -> F1_seen
"""
from __future__ import annotations

import random
import re
from collections import Counter, defaultdict

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from .schema import entity_text

SPLITS = ("train", "val", "seen_test", "unseen_test")


# --------------------------------------------------------------------- fingerprints
def layout_vector(doc: dict, grid: int = 12) -> np.ndarray:
    W, H = max(doc["width"], 1), max(doc["height"], 1)
    g = np.zeros((grid, grid), dtype=np.float64)
    cw, ch = W / grid, H / grid
    for w in doc["words"]:
        x0, y0, x1, y1 = w["box"]
        if x1 <= x0 or y1 <= y0:
            continue
        for gx in range(int(x0 // cw), min(grid - 1, int(x1 // cw)) + 1):
            for gy in range(int(y0 // ch), min(grid - 1, int(y1 // ch)) + 1):
                ox = max(0.0, min(x1, (gx + 1) * cw) - max(x0, gx * cw))
                oy = max(0.0, min(y1, (gy + 1) * ch) - max(y0, gy * ch))
                g[gy, gx] += ox * oy / (cw * ch)
    v = np.sqrt(np.clip(g, 0, None)).ravel()  # sqrt damps dense text blocks
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


_TOKEN = re.compile(r"[^\W\d_]{2,}", re.UNICODE)


def key_text(doc: dict) -> str:
    parts = [entity_text(doc, e) for e in doc["entities"] if e["role"] in ("key", "header")]
    return " ".join(_TOKEN.findall(" ".join(parts).lower()))


def text_matrix(docs: list[dict]) -> np.ndarray:
    """Plain TF-IDF (unigrams + bigrams) without sklearn, L2-normalised rows."""
    toks = []
    for d in docs:
        t = key_text(d).split()
        toks.append(t + [a + "_" + b for a, b in zip(t, t[1:])])
    df = Counter()
    for t in toks:
        df.update(set(t))
    vocab = {w: i for i, w in enumerate(sorted(df))}
    n = len(docs)
    m = np.zeros((n, max(1, len(vocab))), dtype=np.float64)
    for r, t in enumerate(toks):
        for w, c in Counter(t).items():
            m[r, vocab[w]] = (1 + np.log(c)) * (np.log((1 + n) / (1 + df[w])) + 1)
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return m / norms


def distance_matrix(docs: list[dict], alpha: float = 0.5, grid: int = 12) -> np.ndarray:
    L = np.stack([layout_vector(d, grid) for d in docs])
    d_layout = 1.0 - np.clip(L @ L.T, -1, 1)
    if alpha >= 1.0:
        D = d_layout
    else:
        T = text_matrix(docs)
        d_text = 1.0 - np.clip(T @ T.T, -1, 1)
        # documents with no key text at all: fall back to layout only
        empty = np.linalg.norm(T, axis=1) == 0
        d_text[empty, :] = d_layout[empty, :]
        d_text[:, empty] = d_layout[:, empty]
        D = alpha * d_layout + (1 - alpha) * d_text
    D = (D + D.T) / 2.0
    np.fill_diagonal(D, 0.0)
    return np.clip(D, 0.0, None)


def cluster(D: np.ndarray, threshold: float) -> np.ndarray:
    if len(D) == 1:
        return np.array([0])
    Z = linkage(squareform(D, checks=False), method="average")
    return fcluster(Z, t=threshold, criterion="distance") - 1


def induce_templates(docs: list[dict], source: str, alpha: float, threshold: float, grid: int = 12):
    """Fill template_id for docs of one source. Returns (D, labels, nn_info)."""
    D = distance_matrix(docs, alpha=alpha, grid=grid)
    labels = cluster(D, threshold)
    # stable, readable ids: order clusters by their first document id
    first = {}
    for d, lab in zip(docs, labels):
        first.setdefault(lab, d["doc_id"])
    order = {lab: i for i, lab in enumerate(sorted(first, key=lambda l: first[l]))}
    for d, lab in zip(docs, labels):
        d["template_id"] = f"{source}/t{order[lab]:03d}"
    Dn = D.copy()
    np.fill_diagonal(Dn, np.inf)
    nn = Dn.argmin(axis=1) if len(docs) > 1 else np.array([0])
    nn_info = [(docs[i]["doc_id"], docs[int(nn[i])]["doc_id"], float(Dn[i, nn[i]]) if len(docs) > 1 else float("inf"))
               for i in range(len(docs))]
    return D, labels, nn_info


# ---------------------------------------------------------------------------- split
def split_templates(docs: list[dict], unseen_frac: float, val_frac: float, seen_frac: float, seed: int,
                    forced_unseen=(), forced_val=(), min_unseen_templates: int = 2):
    """Assign every doc of ONE source to a split. Returns {doc_id: split}."""
    rng = random.Random(seed)
    by_t = defaultdict(list)
    for d in docs:
        by_t[d["template_id"]].append(d)
    tids = sorted(by_t)
    missing = [t for t in list(forced_unseen) + list(forced_val) if t not in by_t]
    if missing:
        raise ValueError(f"forced templates not found: {missing}. Known: {tids[:20]}")
    n = len(docs)
    unseen, val = list(forced_unseen), list(forced_val)
    pool = [t for t in tids if t not in unseen and t not in val]
    rng.shuffle(pool)
    if not forced_unseen and (unseen_frac > 0 or min_unseen_templates > 0):
        while pool and (sum(len(by_t[t]) for t in unseen) < unseen_frac * n or len(unseen) < min_unseen_templates):
            unseen.append(pool.pop())
    if not forced_val:
        while pool and len(pool) > 1 and (sum(len(by_t[t]) for t in val) < val_frac * n or not val):
            val.append(pool.pop())
    train_t = pool
    if not train_t:
        raise ValueError("no templates left for training - lower unseen/val fractions")
    assign = {}
    for t in unseen:
        for d in by_t[t]:
            assign[d["doc_id"]] = "unseen_test"
    for t in val:
        for d in by_t[t]:
            assign[d["doc_id"]] = "val"
    for t in sorted(train_t):
        ds = sorted(by_t[t], key=lambda d: d["doc_id"])
        rng.shuffle(ds)
        k = 0
        if len(ds) >= 2 and seen_frac > 0:
            k = min(len(ds) - 1, max(1, int(round(seen_frac * len(ds)))))
        for i, d in enumerate(ds):
            assign[d["doc_id"]] = "seen_test" if i < k else "train"
    return assign


def check_split(docs: list[dict], assign: dict) -> dict:
    """Hard assertions for the zero-same-template rule. Raises on any violation."""
    t_of = {d["doc_id"]: d["template_id"] for d in docs}
    by_split = defaultdict(set)
    for doc_id, s in assign.items():
        assert s in SPLITS, f"bad split {s}"
        by_split[s].add(t_of[doc_id])
    train_t = by_split["train"]
    leak_u = train_t & by_split["unseen_test"]
    leak_v = train_t & by_split["val"]
    leak_uv = by_split["val"] & by_split["unseen_test"]
    assert not leak_u, f"LEAK: templates in train and unseen_test: {sorted(leak_u)[:5]}"
    assert not leak_v, f"LEAK: templates in train and val: {sorted(leak_v)[:5]}"
    assert not leak_uv, f"LEAK: templates in val and unseen_test: {sorted(leak_uv)[:5]}"
    stray = by_split["seen_test"] - train_t
    assert not stray, f"seen_test templates missing from train: {sorted(stray)[:5]}"
    assert len(assign) == len(docs), "some documents were not assigned"
    return {s: len(by_split[s]) for s in SPLITS}


def near_duplicates(docs: list[dict], D: np.ndarray, assign: dict, dup_threshold: float):
    """Unseen/val docs whose nearest TRAIN doc is closer than dup_threshold - possible
    template leakage that clustering missed. Reported, never silently fixed."""
    idx = {d["doc_id"]: i for i, d in enumerate(docs)}
    train_idx = [idx[d["doc_id"]] for d in docs if assign[d["doc_id"]] == "train"]
    out = []
    if not train_idx:
        return out
    for d in docs:
        if assign[d["doc_id"]] in ("unseen_test", "val"):
            i = idx[d["doc_id"]]
            j = min(train_idx, key=lambda t: D[i, t])
            if D[i, j] < dup_threshold:
                out.append((d["doc_id"], docs[j]["doc_id"], float(D[i, j])))
    return out
