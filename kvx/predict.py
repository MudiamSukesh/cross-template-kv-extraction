"""Inference on one document: tags, entities, links - in two modes at once.

gold mode : link head runs on the GOLD key/value entities -> isolates link reasoning (link F1)
e2e mode  : entities come from the predicted tags, links from the link head on those
            predicted entities -> what a user actually gets (end-to-end pair F1)
"""
from __future__ import annotations

import contextlib

import torch

from .data import collate
from .encoding import doc_link_entities, doc_units
from .schema import TAGS, iso_box, union_box
from .viz import page_image


def bio_decode(seq_words: list[int], word_tag: dict) -> list[dict]:
    """Lenient BIO: an I- tag continues the open entity only if it has the same role;
    otherwise it starts a new entity (a common, defensible choice)."""
    ents, cur = [], None
    for w in seq_words:
        t = TAGS[word_tag[w]]
        if t == "O":
            cur = None
            continue
        pref, role = t.split("-", 1)
        role = role.lower()
        if pref == "B" or cur is None or cur["role"] != role:
            cur = {"role": role, "word_ids": [w]}
            ents.append(cur)
        else:
            cur["word_ids"].append(w)
    return ents


def _amp(device, dtype):
    if dtype is None or device.type != "cuda":
        return contextlib.nullcontext()
    return torch.autocast(device_type="cuda", dtype=dtype)


def _entity_inputs(enc: dict, doc: dict, ents: list[dict]):
    tok_of = {}
    for t, w in enumerate(enc["token_word"]):
        if w >= 0:
            tok_of.setdefault(w, []).append(t)
    keep, ent_tok, ent_box = [], [], []
    for i, e in enumerate(ents):
        toks = [t for w in e["word_ids"] for t in tok_of.get(w, [])]
        if not toks:
            continue
        keep.append(i)
        ent_tok.append(sorted(toks))
        ent_box.append(union_box(iso_box(doc["words"][w]["box"], doc["width"], doc["height"]) for w in e["word_ids"]))
    return keep, ent_tok, ent_box


def _decode_links(logits: torch.Tensor, keys: list, values: list, top: int = 3):
    """-> list of (value_local, key_local or None, p_choice, [(key_local, p), ...])"""
    probs = torch.softmax(logits.float(), dim=1).cpu()
    out = []
    for i, v in enumerate(values):
        row = probs[i]
        j = int(row.argmax())
        k = keys[j] if j < len(keys) else None
        order = torch.argsort(row, descending=True)[:top].tolist()
        tops = [(keys[c] if c < len(keys) else None, float(row[c])) for c in order]
        out.append((v, k, float(row[j]), tops))
    return out


@torch.no_grad()
def predict_doc(model, encoder, doc: dict, device, amp_dtype=None,
                use_gold_units: bool = True, image=None) -> dict:
    model.eval()
    has_gold = bool(doc.get("entities"))
    units = doc_units(doc, use_gold=use_gold_units and has_gold)
    chunks = encoder.chunk_units(doc, units)
    if image is None and encoder.image_processor is not None:
        image = page_image(doc)
    gold_ents = doc_link_entities(doc) if has_gold else []
    word_tag, word_conf = {}, {}
    pred_entities, pred_links, link_p = [], [], []
    gold_mode = {}   # value doc id -> {"key": key doc id | None, "p": float, "top": [...]}
    lost = 0
    for ch in chunks:
        enc = encoder.encode(doc, ch, gold_ents, image=image)
        lost += len(enc["lost_words"])
        b = collate([enc], encoder.pad_id)
        with _amp(device, amp_dtype):
            seq = model.encode(b["input_ids"].to(device), b["bbox"].to(device), b["attention_mask"].to(device),
                               b["pixel_values"].to(device) if "pixel_values" in b else None)
            probs = torch.softmax(model.tag_logits(seq)[0].float(), dim=-1)
            conf, tags = probs.max(-1)
            tags, conf = tags.tolist(), conf.tolist()
        for p, t in enumerate(enc["first_tok"]):
            if t >= 0:
                word_tag[enc["seq_words"][p]] = tags[t]
                word_conf[enc["seq_words"][p]] = conf[t]
            else:
                word_tag[enc["seq_words"][p]] = 0
                word_conf[enc["seq_words"][p]] = 0.0
        lt = enc["link"]
        if has_gold and lt.values:
            with _amp(device, amp_dtype):
                lg = model.link_logits(seq[0], enc["ent_tok"], enc["ent_box"], lt.keys, lt.values, enc["scale"])
            for v, k, p, tops in _decode_links(lg, lt.keys, lt.values):
                gold_mode[lt.ent_doc_ids[v]] = {
                    "key": lt.ent_doc_ids[k] if k is not None else None, "p": p,
                    "top": [(lt.ent_doc_ids[c] if c is not None else None, q) for c, q in tops]}
        # end-to-end
        ents = bio_decode(enc["seq_words"], word_tag)
        keep, ent_tok, ent_box = _entity_inputs(enc, doc, ents)
        ents = [ents[i] for i in keep]
        base = len(pred_entities)
        pred_entities += ents
        keys = [i for i, e in enumerate(ents) if e["role"] == "key"]
        vals = [i for i, e in enumerate(ents) if e["role"] == "value"]
        if vals:
            with _amp(device, amp_dtype):
                lg = model.link_logits(seq[0], ent_tok, ent_box, keys, vals, enc["scale"])
            for v, k, p, _ in _decode_links(lg, keys, vals):
                if k is not None:
                    pred_links.append([base + k, base + v])
                    link_p.append(round(p, 4))
    gold_mode_links = sorted([info["key"], v] for v, info in gold_mode.items() if info["key"] is not None)
    return {
        "doc_id": doc["doc_id"],
        "n_chunks": len(chunks),
        "lost_words": lost,
        "word_tags": [word_tag.get(i, 0) for i in range(len(doc["words"]))],
        "word_conf": [round(word_conf.get(i, 0.0), 4) for i in range(len(doc["words"]))],
        "pred_entities": pred_entities,
        "pred_links": pred_links,
        "pred_link_p": link_p,
        "gold_mode": {str(k): v for k, v in gold_mode.items()},
        "gold_mode_links": gold_mode_links,
    }
