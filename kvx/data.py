"""Torch dataset and collate function for training."""
from __future__ import annotations

import random

import torch
from torch.utils.data import Dataset

from .encoding import Encoder, doc_link_entities, doc_units, gold_word_tags
from .schema import median_word_height
from .viz import page_image


class ChunkDataset(Dataset):
    """One item = one <=512-token chunk of one document."""

    def __init__(self, docs: list[dict], encoder: Encoder, aug_cfg: dict | None = None, seed: int = 0):
        self.docs = docs
        self.enc = encoder
        self.aug = aug_cfg if (aug_cfg and aug_cfg.get("enabled")) else None
        self.rng = random.Random(seed)
        self.tags = [gold_word_tags(d) for d in docs]
        # no link supervision (SROIE): gold None -> the link head is not trained on that doc
        self.gold = [set(map(tuple, d["links"])) if d.get("link_supervision", True) else None for d in docs]
        self.ents = [doc_link_entities(d) for d in docs]
        self.items = []
        self.item_source = []
        self.n_cross_chunk = 0
        for di, d in enumerate(docs):
            chunks = encoder.chunk_units(d, doc_units(d, use_gold=True))
            for ch in chunks:
                self.items.append((di, ch))
                self.item_source.append(d["source"])
            if len(chunks) > 1:
                where = {}
                for ci, ch in enumerate(chunks):
                    for u in ch:
                        for w in u:
                            where[w] = ci
                ent_chunk = {e["id"]: {where[w] for w in e["word_ids"]} for e in d["entities"]}
                for k, v in d["links"]:
                    if not (ent_chunk[k] & ent_chunk[v]):
                        self.n_cross_chunk += 1

    def source_balanced_weights(self) -> list[float]:
        """Sampling weight per chunk so every source contributes equally per epoch
        (otherwise ~2,700 receipts drown 199 FUNSD forms)."""
        from collections import Counter
        n = Counter(self.item_source)
        return [1.0 / n[s] for s in self.item_source]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        di, ch = self.items[i]
        d = self.docs[di]
        boxes, image, aff = None, None, None
        if self.aug:
            boxes, aff = Encoder.augment([w["box"] for w in d["words"]], d["width"], d["height"],
                                         median_word_height(d), self.aug, self.rng)
        if self.enc.image_processor is not None:
            image = page_image(d)
            if aff is not None:
                image = Encoder.affine_image(image, *aff)
        return self.enc.encode(d, ch, self.ents[di], gold_links=self.gold[di], word_tags=self.tags[di],
                               boxes_px=boxes, image=image)


def collate(batch: list[dict], pad_id: int) -> dict:
    L = max(len(b["input_ids"]) for b in batch)
    n = len(batch)
    input_ids = torch.full((n, L), pad_id, dtype=torch.long)
    attn = torch.zeros((n, L), dtype=torch.long)
    bbox = torch.zeros((n, L, 4), dtype=torch.long)
    labels = torch.full((n, L), -100, dtype=torch.long)
    for i, b in enumerate(batch):
        m = len(b["input_ids"])
        input_ids[i, :m] = torch.as_tensor(b["input_ids"])
        attn[i, :m] = torch.as_tensor(b["attention_mask"])
        bbox[i, :m] = torch.as_tensor(b["bbox"])
        labels[i, :m] = torch.as_tensor(b["labels"])
    out = {"input_ids": input_ids, "attention_mask": attn, "bbox": bbox, "labels": labels,
           "items": batch}
    if "pixel_values" in batch[0]:
        out["pixel_values"] = torch.stack([b["pixel_values"] for b in batch])
    return out
