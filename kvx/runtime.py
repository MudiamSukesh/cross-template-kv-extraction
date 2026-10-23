"""Loading tokenizer / image processor / model, device and precision, checkpoints.

A run directory is self-contained: after training it holds the tokenizer, image processor,
backbone config and weights, so evaluation and the interface work offline."""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import torch

from .encoding import Encoder
from .model import KVModel


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def amp_dtype(mode: str, device: torch.device):
    if device.type != "cuda" or mode == "none":
        return None
    if mode == "bf16" or (mode == "auto" and torch.cuda.is_bf16_supported()):
        return torch.bfloat16
    return torch.float16


def load_tokenizer(path: str):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path)
    probe = tok(["probe"], boxes=[[0, 0, 0, 0]], add_special_tokens=False)
    if "bbox" not in probe:
        raise RuntimeError(f"{path}: tokenizer does not accept boxes - is this a LayoutLMv3 checkpoint?")
    return tok


def load_image_processor(path: str):
    from transformers import LayoutLMv3ImageProcessor
    return LayoutLMv3ImageProcessor.from_pretrained(path, apply_ocr=False)


def build_encoder(cfg: dict, tok_path: str | None = None) -> Encoder:
    path = tok_path or cfg["model"]["backbone"]
    tok = load_tokenizer(path)
    ip = load_image_processor(path) if cfg["model"]["visual"] else None
    return Encoder(tok, cfg["model"]["max_tokens"], cfg["model"]["box_level"], ip)


def save_run_assets(run_dir: Path, encoder: Encoder, model: KVModel):
    assets = run_dir / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    encoder.tok.save_pretrained(assets)
    if encoder.image_processor is not None:
        encoder.image_processor.save_pretrained(assets)
    with open(assets / "backbone_config.json", "w", encoding="utf-8") as f:
        json.dump(model.backbone.config.to_dict(), f)


def save_checkpoint(path: Path, model: KVModel, cfg: dict, epoch: int, val: dict):
    tmp = Path(str(path) + ".tmp")
    torch.save({"model": model.state_dict(), "config": cfg, "epoch": epoch, "val": val}, tmp)
    os.replace(tmp, path)


def load_run(run_dir: str | Path, device=None, checkpoint: str = "best.pt"):
    """-> (model, encoder, cfg, amp_dtype). Works offline from the run directory."""
    from transformers import LayoutLMv3Config, LayoutLMv3Model
    run_dir = Path(run_dir)
    device = device or get_device()
    ck = torch.load(run_dir / checkpoint, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    assets = run_dir / "assets"
    with open(assets / "backbone_config.json", encoding="utf-8") as f:
        bcfg = LayoutLMv3Config.from_dict(json.load(f))
    model = KVModel(cfg["model"], backbone=LayoutLMv3Model(bcfg))
    missing, unexpected = model.load_state_dict(ck["model"], strict=False)
    bad = [k for k in missing if "position_ids" not in k] + [k for k in unexpected if "position_ids" not in k]
    if bad:
        raise RuntimeError(f"checkpoint does not match model: {bad[:5]}")
    model.to(device).eval()
    tok = load_tokenizer(str(assets))
    ip = load_image_processor(str(assets)) if cfg["model"]["visual"] else None
    enc = Encoder(tok, cfg["model"]["max_tokens"], cfg["model"]["box_level"], ip)
    return model, enc, cfg, amp_dtype(cfg["train"]["amp"], device), ck
