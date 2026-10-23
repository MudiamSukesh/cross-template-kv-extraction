"""Turn a unified document into LayoutLMv3 inputs.

Sequence order
  The document is cut into *units* - gold entities when annotations exist, OCR lines
  for uploaded scans, single words otherwise - placed in reading order. Units are kept
  whole inside a chunk so a BIO span is never broken across chunks.
  NOTE (reported as a limitation): with gold units the words of one entity are
  contiguous, which is slightly optimistic compared with raw OCR order. This is the
  standard FUNSD protocol, and the failure analysis measures the effect directly.

Chunking
  LayoutLMv3 takes 512 tokens. Longer pages are split into several chunks along unit
  boundaries. A gold link whose key and value land in different chunks cannot be
  predicted; the evaluation counts such links as misses (never silently drops them).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from PIL import Image

from .schema import (TAG2ID, ROLE2TAG, TAGGED_ROLES, iso_box, median_word_height, normalize_box,
                     reading_order_key, union_box)


@dataclass
class LinkTargets:
    roles: list = field(default_factory=list)      # role per local entity
    ent_doc_ids: list = field(default_factory=list)  # doc-level entity id (or -1 for predicted)
    keys: list = field(default_factory=list)       # local indices of key entities
    values: list = field(default_factory=list)     # local indices of value entities
    targets: list = field(default_factory=list)    # per value: list of positions in `keys`; [] = no key; None = unknown


def word_scale(iso_boxes) -> float:
    """Median word height in the iso frame - the unit of all relative link geometry."""
    hs = sorted(b[3] - b[1] for b in iso_boxes if b[3] > b[1])
    return max(float(hs[len(hs) // 2]), 1.0) if hs else 10.0


def gold_word_tags(doc: dict) -> list[int]:
    """Per-word tag ids; -100 for words without supervision (doc["ignore_words"])."""
    tags = [TAG2ID["O"]] * len(doc["words"])
    for e in doc["entities"]:
        if e["role"] in TAGGED_ROLES:
            t = ROLE2TAG[e["role"]]
            for j, w in enumerate(e["word_ids"]):
                tags[w] = TAG2ID[("B-" if j == 0 else "I-") + t]
    for w in doc.get("ignore_words", []) or []:
        tags[w] = -100
    return tags


def doc_units(doc: dict, use_gold: bool = True) -> list[list[int]]:
    """Ordered list of word-id groups the sequence is built from.
    1. doc["lines"]  - OCR-style text segments (SROIE/WildReceipt boxes, VRDU OCR lines, docTR lines);
                       their own order is kept (OCR reading order)
    2. gold entities - FUNSD, which has no separate line annotation (only when use_gold)
    3. single words  - sorted top-to-bottom, left-to-right"""
    tol = median_word_height(doc)
    words = doc["words"]
    if doc.get("lines"):
        units = [list(l) for l in doc["lines"] if l]
        covered = {w for u in units for w in u}
        extra = [[i] for i in range(len(words)) if i not in covered]
        if extra:
            units += extra
            units.sort(key=lambda u: reading_order_key(union_box(words[i]["box"] for i in u), tol))
        return units
    if use_gold and doc.get("entities"):
        units = [list(e["word_ids"]) for e in doc["entities"]]
    else:
        units = [[i] for i in range(len(words))]
    covered = {w for u in units for w in u}
    units += [[i] for i in range(len(words)) if i not in covered]
    units.sort(key=lambda u: reading_order_key(union_box(words[i]["box"] for i in u), tol))
    return units


class Encoder:
    def __init__(self, tokenizer, max_tokens: int = 512, box_level: str = "word", image_processor=None):
        self.tok = tokenizer
        self.max_tokens = max_tokens
        self.box_level = box_level
        self.image_processor = image_processor
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 1

    # ------------------------------------------------------------------ chunking
    def token_counts(self, texts: list[str]) -> list[int]:
        if not texts:
            return []
        enc = self.tok(texts, boxes=[[0, 0, 0, 0]] * len(texts), add_special_tokens=False,
                       truncation=False, return_attention_mask=False)
        counts = [0] * len(texts)
        for wid in enc.word_ids():
            if wid is not None:
                counts[wid] += 1
        return [max(1, c) for c in counts]

    def chunk_units(self, doc: dict, units: list[list[int]]) -> list[list[list[int]]]:
        counts = self.token_counts([w["text"] for w in doc["words"]])
        budget = self.max_tokens - 2
        chunks, cur, used = [], [], 0
        for u in units:
            need = sum(counts[w] for w in u)
            if need > budget:  # a single giant unit: split it word-wise
                if cur:
                    chunks.append(cur)
                    cur, used = [], 0
                piece, pu = [], 0
                for w in u:
                    if pu + counts[w] > budget and piece:
                        chunks.append([piece])
                        piece, pu = [], 0
                    piece.append(w)
                    pu += counts[w]
                if piece:
                    cur, used = [piece], pu
                continue
            if used + need > budget and cur:
                chunks.append(cur)
                cur, used = [], 0
            cur.append(u)
            used += need
        if cur:
            chunks.append(cur)
        return chunks

    # -------------------------------------------------------------- augmentation
    @staticmethod
    def augment(boxes, W, H, med_h, cfg, rng: random.Random):
        s = rng.uniform(cfg["scale_min"], 1.0) if cfg["scale_min"] < 1.0 else 1.0
        tx = rng.uniform(0, (1 - s) * W) if cfg["shift"] else 0.0
        ty = rng.uniform(0, (1 - s) * H) if cfg["shift"] else 0.0
        j = cfg["jitter"] * med_h * s
        out = []
        for b in boxes:
            dx = rng.uniform(-j, j) if j > 0 else 0.0
            dy = rng.uniform(-j, j) if j > 0 else 0.0
            nb = [b[0] * s + tx + dx, b[1] * s + ty + dy, b[2] * s + tx + dx, b[3] * s + ty + dy]
            out.append([min(max(nb[0], 0), W), min(max(nb[1], 0), H), min(max(nb[2], 0), W), min(max(nb[3], 0), H)])
        return out, (s, tx, ty)

    @staticmethod
    def affine_image(im: Image.Image, s, tx, ty) -> Image.Image:
        if s == 1.0 and tx == 0 and ty == 0:
            return im
        return im.transform(im.size, Image.AFFINE, data=(1 / s, 0, -tx / s, 0, 1 / s, -ty / s),
                            resample=Image.BILINEAR, fillcolor=(255, 255, 255))

    # ------------------------------------------------------------------ encoding
    def encode(self, doc: dict, chunk: list[list[int]], link_entities: list[tuple], gold_links=None,
               word_tags=None, boxes_px=None, image: Image.Image | None = None) -> dict:
        """
        chunk          : list of units (word-id lists) in sequence order
        link_entities  : [(role, word_ids, doc_entity_id), ...] candidates for the link head
        gold_links     : set of (key_doc_id, value_doc_id) or None (inference)
        word_tags      : per-doc-word tag ids or None (inference)
        boxes_px       : per-doc-word pixel boxes (augmented) or None -> doc boxes
        """
        W, H = doc["width"], doc["height"]
        boxes_px = boxes_px if boxes_px is not None else [w["box"] for w in doc["words"]]
        seq = [w for u in chunk for w in u]
        pos_of = {w: p for p, w in enumerate(seq)}
        if self.box_level == "segment":
            seg = {}
            for u in chunk:
                ub = union_box(boxes_px[w] for w in u)
                for w in u:
                    seg[w] = ub
            boxes = [normalize_box(seg[w], W, H) for w in seq]
        else:
            boxes = [normalize_box(boxes_px[w], W, H) for w in seq]
        texts = [doc["words"][w]["text"] for w in seq]
        enc = self.tok(texts, boxes=boxes, truncation=True, max_length=self.max_tokens,
                       return_attention_mask=True)
        wids = enc.word_ids()
        first_tok = {}
        for t, p in enumerate(wids):
            if p is not None and p not in first_tok:
                first_tok[p] = t
        lost = [seq[p] for p in range(len(seq)) if p not in first_tok]
        labels = [-100] * len(wids)
        if word_tags is not None:
            for p, t in first_tok.items():
                labels[t] = word_tags[seq[p]]
        tok_lists = {}
        for t, p in enumerate(wids):
            if p is not None:
                tok_lists.setdefault(seq[p], []).append(t)

        # link-head entities fully present in this chunk
        lt = LinkTargets()
        ent_tok, ent_box = [], []
        for role, wlist, did in link_entities:
            if role not in ("key", "value"):
                continue
            if not all(w in pos_of and w in tok_lists for w in wlist):
                continue
            idx = len(lt.roles)
            lt.roles.append(role)
            lt.ent_doc_ids.append(did)
            toks = sorted(t for w in wlist for t in tok_lists[w])
            ent_tok.append(toks)
            ent_box.append(union_box(iso_box(boxes_px[w], W, H) for w in wlist))
            (lt.keys if role == "key" else lt.values).append(idx)
        if gold_links is not None:
            key_pos = {lt.ent_doc_ids[k]: i for i, k in enumerate(lt.keys)}
            gold_keys_of = {}
            for k, v in gold_links:
                gold_keys_of.setdefault(v, []).append(k)
            for v in lt.values:
                gk = gold_keys_of.get(lt.ent_doc_ids[v], [])
                present = [key_pos[k] for k in gk if k in key_pos]
                if gk and not present:
                    lt.targets.append(None)  # gold key lives in another chunk
                else:
                    lt.targets.append(present)
        scale = word_scale([iso_box(boxes_px[w], W, H) for w in seq])

        out = {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "bbox": enc["bbox"],
            "labels": labels,
            "token_word": [seq[p] if p is not None else -1 for p in wids],
            "first_tok": [first_tok.get(p, -1) for p in range(len(seq))],
            "seq_words": seq,
            "lost_words": lost,
            "ent_tok": ent_tok,
            "ent_box": ent_box,
            "link": lt,
            "scale": max(scale, 1.0),
        }
        if self.image_processor is not None:
            im = image if image is not None else Image.new("RGB", (int(W), int(H)), "white")
            out["pixel_values"] = self.image_processor(im, return_tensors="pt")["pixel_values"][0]
        return out


def doc_link_entities(doc: dict):
    return [(e["role"], e["word_ids"], e["id"]) for e in doc["entities"]]
