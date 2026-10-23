"""Configuration: defaults + YAML overrides. Unknown keys are an error (typos must not
silently fall back to defaults - that is how ablation tables go wrong)."""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

DEFAULTS = {
    "data": {
        "docs": "cache/split/docs_split.jsonl",
        "train_sources": [],       # empty = all sources in the split file
        "eval_sources": [],
        "test_docs": [],           # extra held-out files (VRDU), streamed at evaluation, never trained on
        "balance_sources": True,   # sample every training source equally per epoch
    },
    "model": {
        "backbone": "microsoft/layoutlmv3-base",
        "visual": False,           # pixels carry template style; ablate with true
        "box_level": "word",       # word | segment (segment = gold entity box, optimistic)
        "max_tokens": 512,
        "dropout": 0.1,
        "ent_dim": 256,
        "geom_dim": 64,
        "pair_hidden": 512,
        "link_geometry": "relative",  # relative | absolute | none  <- ablation
    },
    "train": {
        "epochs": 20,
        "batch_size": 2,
        "grad_accum": 2,
        "lr": 3.0e-5,
        "head_lr": 3.0e-4,
        "weight_decay": 0.01,
        "warmup_ratio": 0.1,
        "link_weight": 1.0,
        "max_grad_norm": 1.0,
        "amp": "auto",             # auto | bf16 | fp16 | none
        "select_metric": "mean",   # on VAL TEMPLATES only: mean of value recall, entity, link, e2e pair F1
        "patience": 5,
        "seed": 0,
        "num_workers": 0,          # keep 0 on Windows
        "out_dir": "runs/base",
    },
    "augment": {
        "enabled": True,
        "scale_min": 0.85,         # global page scale
        "shift": True,             # global translation (keeps page inside the grid)
        "jitter": 0.15,            # per-word box jitter, in units of median word height
    },
    "eval": {
        "splits": ["val", "seen_test", "unseen_test"],
        "bootstrap": 1000,
        "fuzzy_threshold": 0.8,
        "n_qualitative": 12,
    },
}


def _merge(base: dict, over: dict, path: str = ""):
    for k, v in over.items():
        if k not in base:
            raise KeyError(f"unknown config key: {path}{k}")
        if isinstance(base[k], dict):
            if not isinstance(v, dict):
                raise TypeError(f"config key {path}{k} must be a mapping")
            _merge(base[k], v, path + k + ".")
        else:
            base[k] = v


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> dict:
    cfg = copy.deepcopy(DEFAULTS)
    if path:
        with open(path, encoding="utf-8") as f:
            _merge(cfg, yaml.safe_load(f) or {})
    for ov in overrides or []:  # "train.epochs=5"
        key, _, val = ov.partition("=")
        parts = key.strip().split(".")
        node = {}
        cur = node
        for p in parts[:-1]:
            cur[p] = {}
            cur = cur[p]
        cur[parts[-1]] = yaml.safe_load(val)
        _merge(cfg, node)
    check_config(cfg)
    return cfg


def check_config(cfg: dict):
    m = cfg["model"]
    assert m["box_level"] in ("word", "segment"), "model.box_level must be word|segment"
    assert m["link_geometry"] in ("relative", "absolute", "none"), "model.link_geometry must be relative|absolute|none"
    assert 64 <= m["max_tokens"] <= 512, "model.max_tokens must be in [64, 512]"
    assert cfg["train"]["amp"] in ("auto", "bf16", "fp16", "none")
    assert cfg["train"]["select_metric"] in ("mean", "value_recall", "e2e_pair_f1", "link_f1", "entity_f1")


def save_config(cfg: dict, path: str | Path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
