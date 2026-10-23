"""Training loop. Model selection uses the VAL templates (held out from training), never
seen_test or unseen_test - otherwise early stopping quietly rewards template memorisation."""
from __future__ import annotations

import json
import math
import time
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from .config import save_config
from .data import ChunkDataset, collate
from .metrics import doc_counts, fmt, summarize
from .model import KVModel, count_parameters, link_loss, tag_loss
from .predict import predict_doc
from .runtime import (amp_dtype, build_encoder, get_device, save_checkpoint, save_run_assets,
                      seed_everything)
from .schema import read_jsonl


def select_docs(docs, split, sources):
    return [d for d in docs if d.get("split") == split and (not sources or d["source"] in sources)]


def evaluate_docs(model, encoder, docs, device, adt, fuzzy=0.8):
    preds, counts = {}, {}
    for d in docs:
        p = predict_doc(model, encoder, d, device, adt)
        preds[d["doc_id"]] = p
        counts[d["doc_id"]] = doc_counts(d, p, fuzzy)
    return preds, counts


def _param_groups(model: KVModel, lr, head_lr, wd):
    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight", "norm.weight")
    groups = {("b", True): [], ("b", False): [], ("h", True): [], ("h", False): []}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        part = "b" if n.startswith("backbone.") else "h"
        groups[(part, not any(n.endswith(x) for x in no_decay))].append(p)
    out = []
    for (part, decay), ps in groups.items():
        if ps:
            out.append({"params": ps, "lr": lr if part == "b" else head_lr, "weight_decay": wd if decay else 0.0})
    return out


def train(cfg: dict, log=print, backbone=None) -> Path:
    seed_everything(cfg["train"]["seed"])
    device = get_device()
    adt = amp_dtype(cfg["train"]["amp"], device)
    run_dir = Path(cfg["train"]["out_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, run_dir / "config.yaml")

    docs = read_jsonl(cfg["data"]["docs"])
    if "split" not in docs[0]:
        raise SystemExit(f"{cfg['data']['docs']} has no 'split' field - run scripts/02_template_split.py first")
    src = cfg["data"]["train_sources"]
    train_docs = select_docs(docs, "train", src)
    val_docs = select_docs(docs, "val", src)
    if not train_docs or not val_docs:
        raise SystemExit(f"need train and val documents (got {len(train_docs)} / {len(val_docs)})")
    # zero-same-template guard, repeated here so a hand-edited split file cannot slip through
    tr_t = {d["template_id"] for d in train_docs}
    held = {d["template_id"] for d in docs if d.get("split") in ("val", "unseen_test")}
    assert not (tr_t & held), f"LEAK: {sorted(tr_t & held)[:5]}"

    encoder = build_encoder(cfg)
    model = KVModel(cfg["model"], backbone=backbone).to(device)
    save_run_assets(run_dir, encoder, model)
    ds = ChunkDataset(train_docs, encoder, cfg["augment"], seed=cfg["train"]["seed"])
    sampler = None
    if cfg["data"]["balance_sources"] and len(set(ds.item_source)) > 1:
        sampler = WeightedRandomSampler(ds.source_balanced_weights(), num_samples=len(ds), replacement=True,
                                        generator=torch.Generator().manual_seed(cfg["train"]["seed"]))
    loader = DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=sampler is None, sampler=sampler,
                        num_workers=cfg["train"]["num_workers"],
                        collate_fn=partial(collate, pad_id=encoder.pad_id))
    log(f"device={device} amp={adt} | {count_parameters(model)}")
    from collections import Counter
    log(f"train: {len(train_docs)} docs / {len(ds)} chunks / {len(tr_t)} templates | "
        f"val: {len(val_docs)} docs / {len({d['template_id'] for d in val_docs})} templates")
    log("train docs per source: " + ", ".join(f"{k}={v}" for k, v in sorted(Counter(d['source'] for d in train_docs).items()))
        + (" | source-balanced sampling" if sampler is not None else ""))
    if ds.n_cross_chunk:
        log(f"note: {ds.n_cross_chunk} training links cross chunk boundaries and cannot be learned")

    tc = cfg["train"]
    opt = torch.optim.AdamW(_param_groups(model, tc["lr"], tc["head_lr"], tc["weight_decay"]))
    steps_per_epoch = math.ceil(len(loader) / tc["grad_accum"])
    total = max(1, steps_per_epoch * tc["epochs"])
    warm = int(tc["warmup_ratio"] * total)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(1, warm) if s < warm else max(0.0, (total - s) / max(1, total - warm)))
    scaler = torch.amp.GradScaler("cuda") if adt == torch.float16 else None

    best, best_epoch, bad = -1.0, -1, 0
    hist_path = run_dir / "history.jsonl"
    hist_path.write_text("", encoding="utf-8")
    for epoch in range(1, tc["epochs"] + 1):
        model.train()
        t0, sum_tag, sum_link, n_batches = time.time(), 0.0, 0.0, 0
        opt.zero_grad(set_to_none=True)
        for bi, batch in enumerate(loader):
            ctx = torch.autocast("cuda", dtype=adt) if adt is not None else torch.autocast("cpu", enabled=False)
            with ctx:
                seq = model.encode(batch["input_ids"].to(device), batch["bbox"].to(device),
                                   batch["attention_mask"].to(device),
                                   batch["pixel_values"].to(device) if "pixel_values" in batch else None)
                lt_loss = tag_loss(model.tag_logits(seq), batch["labels"].to(device))
                link_sum, n_link = seq.new_zeros(()).float(), 0
                for i, it in enumerate(batch["items"]):
                    lt = it["link"]
                    if lt.values and any(t is not None for t in lt.targets):
                        lg = model.link_logits(seq[i], it["ent_tok"], it["ent_box"], lt.keys, lt.values, it["scale"])
                        s, k = link_loss(lg, lt.targets)
                        link_sum = link_sum + s
                        n_link += k
                l_link = link_sum / max(n_link, 1)
                loss = (lt_loss + tc["link_weight"] * l_link) / tc["grad_accum"]
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at epoch {epoch} batch {bi}")
            (scaler.scale(loss) if scaler else loss).backward()
            if (bi + 1) % tc["grad_accum"] == 0 or bi + 1 == len(loader):
                if scaler:
                    scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), tc["max_grad_norm"])
                if scaler:
                    scaler.step(opt)
                    scaler.update()
                else:
                    opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
            sum_tag += lt_loss.detach().item()
            sum_link += l_link.detach().item()
            n_batches += 1

        _, counts = evaluate_docs(model, encoder, val_docs, device, adt, cfg["eval"]["fuzzy_threshold"])
        val = summarize(list(counts.values()))
        score = val[tc["select_metric"]] or 0.0
        rec = {"epoch": epoch, "tag_loss": sum_tag / max(1, n_batches), "link_loss": sum_link / max(1, n_batches),
               "val_value_recall": val["value_recall"], "val_entity_f1": val["entity_f1"], "val_link_f1": val["link_f1"],
               "val_e2e_pair_f1": val["e2e_pair_f1"], "select": score, "seconds": round(time.time() - t0, 1)}
        with open(hist_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        flag = ""
        if score > best:
            best, best_epoch, bad = score, epoch, 0
            save_checkpoint(run_dir / "best.pt", model, cfg, epoch, val)
            flag = "  * saved"
        else:
            bad += 1
        log(f"epoch {epoch:3d} | tag {rec['tag_loss']:.4f} link {rec['link_loss']:.4f} | val value-rec "
            f"{fmt(val['value_recall'])} entity {fmt(val['entity_f1'])} link {fmt(val['link_f1'])} "
            f"e2e {fmt(val['e2e_pair_f1'])} | {rec['seconds']}s{flag}")
        if bad >= tc["patience"]:
            log(f"early stop: no val improvement for {bad} epochs")
            break
    log(f"best epoch {best_epoch} ({tc['select_metric']}={best:.4f}) -> {run_dir / 'best.pt'}")
    return run_dir
