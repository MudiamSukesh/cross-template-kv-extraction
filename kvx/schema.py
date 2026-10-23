"""Unified document schema and small geometry helpers shared by every module.

One document = one JSON object (one line of a .jsonl file):

    doc_id        str   unique, prefixed by source, e.g. "funsd/82092117"
    source        str   "funsd" | "xfund" | "custom"
    split_hint    str   the dataset's own split ("train"/"test"/"val"); informational only
    lang          str   ISO language code
    template_id   str|None  filled by kvx.templates (step 2); None before that
    image_path    str|None  absolute path to the page image
    width, height int   page size in pixels
    words         [{"text": str, "box": [x0, y0, x1, y1]}]      pixel coordinates
    entities      [{"id": int, "role": "key"|"value"|"header"|"other",
                    "word_ids": [int, ...]}]                     id == list index
    links         [[key_entity_id, value_entity_id], ...]       key -> value
    header_links  [[header_entity_id, key_entity_id], ...]      kept, not trained on

Optional fields
    annotation    "full"   keys, values and links annotated (FUNSD, WildReceipt)
                  "values" only some values annotated, no keys/links (SROIE, VRDU)
                  "none"   nothing annotated (uploads)
    lines         [[word ids], ...]  OCR-style text segments in reading order; when present they
                  are the units the sequence is built from (kvx.encoding.doc_units)
    ignore_words  word ids with no tag supervision (unlabelled text in "values" docs, WildReceipt Ignore)
    link_supervision  false -> the link head is not trained on this doc
    gold_values   [{"field": str|None, "text": str, "word_ids": [...]?}]  values for text-level metrics
    pdf_path, page_index   page rendered on demand when image_path is absent (VRDU)
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable, Iterator

ROLES = ("key", "value", "header", "other")
TAGGED_ROLES = ("key", "value", "header")  # roles the tagger predicts; "other" -> O
TAGS = ["O", "B-KEY", "I-KEY", "B-VALUE", "I-VALUE", "B-HEADER", "I-HEADER"]
TAG2ID = {t: i for i, t in enumerate(TAGS)}
ROLE2TAG = {"key": "KEY", "value": "VALUE", "header": "HEADER"}
TAG2ROLE = {v: k for k, v in ROLE2TAG.items()}


# ----------------------------------------------------------------------------- io
def read_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def iter_jsonl(path: str | Path) -> Iterator[dict]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: str | Path, docs: Iterable[dict]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
            n += 1
    return n


# ----------------------------------------------------------------------- validity
def validate_doc(doc: dict) -> list[str]:
    """Return a list of problems; empty list means the document is well formed."""
    errs = []
    for k in ("doc_id", "source", "width", "height", "words", "entities", "links"):
        if k not in doc:
            errs.append(f"missing field {k}")
    if errs:
        return errs
    nw = len(doc["words"])
    seen_words: set[int] = set()
    for i, e in enumerate(doc["entities"]):
        if e["id"] != i:
            errs.append(f"entity {i} has id {e['id']}")
        if e["role"] not in ROLES:
            errs.append(f"entity {i} bad role {e['role']}")
        if not e["word_ids"]:
            errs.append(f"entity {i} has no words")
        for w in e["word_ids"]:
            if not 0 <= w < nw:
                errs.append(f"entity {i} word index {w} out of range")
            if w in seen_words:
                errs.append(f"word {w} belongs to two entities")
            seen_words.add(w)
    for lst in doc.get("lines", []) or []:
        for w in lst:
            if not 0 <= w < nw:
                errs.append(f"line word index {w} out of range")
                break
    for w in doc.get("ignore_words", []) or []:
        if not 0 <= w < nw:
            errs.append(f"ignore word index {w} out of range")
            break
    if doc.get("annotation", "full") not in ("full", "values", "none"):
        errs.append(f"bad annotation kind {doc.get('annotation')}")
    roles = [e["role"] for e in doc["entities"]]
    for k, v in doc["links"]:
        if not (0 <= k < len(roles) and 0 <= v < len(roles)):
            errs.append(f"link {k}->{v} out of range")
        elif roles[k] != "key" or roles[v] != "value":
            errs.append(f"link {k}->{v} is {roles[k]}->{roles[v]}, expected key->value")
    for w in doc["words"]:
        x0, y0, x1, y1 = w["box"]
        if x1 < x0 or y1 < y0:
            errs.append(f"inverted box {w['box']}")
            break
    return errs


# ----------------------------------------------------------------------- geometry
def clip_box(b, w: float, h: float) -> list[float]:
    x0, y0, x1, y1 = [float(v) for v in b]
    x0, x1 = sorted((x0, x1))
    y0, y1 = sorted((y0, y1))
    return [max(0.0, min(x0, w)), max(0.0, min(y0, h)), max(0.0, min(x1, w)), max(0.0, min(y1, h))]


def union_box(boxes: Iterable) -> list[float]:
    boxes = list(boxes)
    return [min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes)]


def entity_box(doc: dict, ent: dict) -> list[float]:
    return union_box(doc["words"][i]["box"] for i in ent["word_ids"])


def entity_text(doc: dict, ent: dict) -> str:
    return " ".join(doc["words"][i]["text"] for i in ent["word_ids"])


def centre(b) -> tuple[float, float]:
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


def median_word_height(doc: dict) -> float:
    hs = sorted(w["box"][3] - w["box"][1] for w in doc["words"] if w["box"][3] > w["box"][1])
    if not hs:
        return max(1.0, doc["height"] / 100.0)
    return max(1.0, hs[len(hs) // 2])


def normalize_box(b, w: float, h: float) -> list[int]:
    """Pixel box -> LayoutLMv3's 0..1000 integer grid."""
    w = max(float(w), 1.0)
    h = max(float(h), 1.0)
    out = [int(round(1000 * b[0] / w)), int(round(1000 * b[1] / h)),
           int(round(1000 * b[2] / w)), int(round(1000 * b[3] / h))]
    return [min(1000, max(0, v)) for v in out]


def iso_box(b, w: float, h: float) -> list[float]:
    """Pixel box -> aspect-preserving frame (longest page side = 1000). Used for link geometry so a
    horizontal and a vertical gap of the same physical size get the same value on any paper format.
    (LayoutLMv3's own 0..1000 grid scales x and y separately - fine as encoder input, wrong for distances.)"""
    s = 1000.0 / max(float(w), float(h), 1.0)
    return [b[0] * s, b[1] * s, b[2] * s, b[3] * s]


def reading_order_key(box, line_tol: float):
    """Sort key: top-to-bottom by line band, then left-to-right."""
    cy = (box[1] + box[3]) / 2.0
    return (int(cy // max(line_tol, 1.0)), box[0])


def count_lines(boxes: list, line_tol: float) -> int:
    """Number of distinct text lines among word boxes (greedy y-centre clustering)."""
    if not boxes:
        return 0
    ys = sorted((b[1] + b[3]) / 2.0 for b in boxes)
    lines = 1
    anchor = ys[0]
    for y in ys[1:]:
        if y - anchor > line_tol:
            lines += 1
            anchor = y
    return lines


def overlap_1d(a0, a1, b0, b1) -> float:
    """Overlap length divided by the shorter interval."""
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    short = max(1e-6, min(a1 - a0, b1 - b0))
    return inter / short


def relation_direction(kbox, vbox) -> str:
    """Where the value sits relative to its key: right / below / left / above / diagonal / overlap."""
    vov = overlap_1d(kbox[1], kbox[3], vbox[1], vbox[3])
    hov = overlap_1d(kbox[0], kbox[2], vbox[0], vbox[2])
    if vov > 0.3 and hov > 0.3:
        return "overlap"
    if vov > 0.3:
        return "right" if vbox[0] >= kbox[0] else "left"
    if hov > 0.3:
        return "below" if vbox[1] >= kbox[1] else "above"
    kc, vc = centre(kbox), centre(vbox)
    if vc[1] > kc[1] and vc[0] >= kc[0]:
        return "diagonal-below-right"
    return "diagonal-other"


def edge_gap(kbox, vbox) -> float:
    """Euclidean gap between two boxes (0 if they touch or overlap)."""
    dx = max(0.0, max(kbox[0], vbox[0]) - min(kbox[2], vbox[2]))
    dy = max(0.0, max(kbox[1], vbox[1]) - min(kbox[3], vbox[3]))
    return math.hypot(dx, dy)
