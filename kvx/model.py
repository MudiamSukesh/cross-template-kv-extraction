"""LayoutLMv3 encoder + two heads.

Head 1 - role tagger   : per word BIO over {KEY, VALUE, HEADER}; first sub-token carries the label.
Head 2 - link scorer   : for every VALUE entity v and KEY entity k in the same chunk,
                         s(k, v) = MLP([e_k ; e_v ; e_k * e_v ; phi(g(k, v))])
                         plus a learned "no key" score n(v). Each value takes a softmax over
                         {keys} U {no key}. Training maximises the probability of its gold key(s)
                         (or of "no key"); decoding takes the arg-max. This is exactly the
                         value -> key direction used at inference, so a key may own several
                         values (tables, multi-line answers) without any extra head.

g(k, v) is RELATIVE geometry in units of the median word height: centre offsets, edge gaps,
gap distance, direction, axis overlaps, size ratios. Absolute page position is deliberately
absent (it is how a model memorises a template); `link_geometry: absolute` adds it back as an
ablation and `none` removes geometry entirely.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .schema import TAGS

N_REL = 12


def _slog(x):
    return torch.sign(x) * torch.log1p(torch.abs(x))


def geometry_features(kb: torch.Tensor, vb: torch.Tensor, scale: float, mode: str) -> torch.Tensor:
    """kb [K,4], vb [V,4] boxes in the aspect-preserving frame (kvx.schema.iso_box) -> [V, K, F].
    scale = median word height in the same frame."""
    V, K = vb.size(0), kb.size(0)
    if mode == "none":
        return vb.new_zeros(V, K, 1)
    k = kb.unsqueeze(0).expand(V, K, 4)
    v = vb.unsqueeze(1).expand(V, K, 4)
    s = max(float(scale), 1.0)
    kcx, kcy = (k[..., 0] + k[..., 2]) / 2, (k[..., 1] + k[..., 3]) / 2
    vcx, vcy = (v[..., 0] + v[..., 2]) / 2, (v[..., 1] + v[..., 3]) / 2
    dx, dy = (vcx - kcx) / s, (vcy - kcy) / s
    gap_x = (v[..., 0] - k[..., 2]) / s
    gap_y = (v[..., 1] - k[..., 3]) / s
    ex = torch.clamp(torch.maximum(k[..., 0], v[..., 0]) - torch.minimum(k[..., 2], v[..., 2]), min=0) / s
    ey = torch.clamp(torch.maximum(k[..., 1], v[..., 1]) - torch.minimum(k[..., 3], v[..., 3]), min=0) / s
    dist = torch.log1p(torch.sqrt(ex ** 2 + ey ** 2))
    ang = torch.atan2(dy, dx)
    kw, kh = (k[..., 2] - k[..., 0]).clamp(min=1), (k[..., 3] - k[..., 1]).clamp(min=1)
    vw, vh = (v[..., 2] - v[..., 0]).clamp(min=1), (v[..., 3] - v[..., 1]).clamp(min=1)
    vov = (torch.minimum(k[..., 3], v[..., 3]) - torch.maximum(k[..., 1], v[..., 1])).clamp(min=0) / torch.minimum(kh, vh)
    hov = (torch.minimum(k[..., 2], v[..., 2]) - torch.maximum(k[..., 0], v[..., 0])).clamp(min=0) / torch.minimum(kw, vw)
    feats = [_slog(dx), _slog(dy), _slog(gap_x), _slog(gap_y), dist, torch.sin(ang), torch.cos(ang),
             vov, hov, torch.log(vw / kw), torch.log(vh / kh), _slog((v[..., 0] - k[..., 0]) / s)]
    if mode == "absolute":
        feats += [kcx / 1000, kcy / 1000, vcx / 1000, vcy / 1000]
    return torch.stack(feats, dim=-1)


def n_geom(mode: str) -> int:
    return {"relative": N_REL, "absolute": N_REL + 4, "none": 1}[mode]


class KVModel(nn.Module):
    def __init__(self, mcfg: dict, backbone: nn.Module | None = None):
        super().__init__()
        if backbone is None:
            from transformers import LayoutLMv3Model
            backbone = LayoutLMv3Model.from_pretrained(mcfg["backbone"])
        self.backbone = backbone
        self.mode = mcfg["link_geometry"]
        self.visual = bool(mcfg["visual"])
        H = backbone.config.hidden_size
        E, G, P, p = mcfg["ent_dim"], mcfg["geom_dim"], mcfg["pair_hidden"], mcfg["dropout"]
        self.drop = nn.Dropout(p)
        self.tagger = nn.Linear(H, len(TAGS))
        self.ent = nn.Sequential(nn.Linear(H, E), nn.GELU(), nn.LayerNorm(E))
        self.geom = nn.Sequential(nn.Linear(n_geom(self.mode), G), nn.GELU(), nn.Linear(G, G))
        self.pair = nn.Sequential(nn.Linear(3 * E + G, P), nn.GELU(), nn.Dropout(p), nn.Linear(P, 1))
        self.null = nn.Sequential(nn.Linear(E, E), nn.GELU(), nn.Linear(E, 1))

    def head_parameters(self):
        for n, p in self.named_parameters():
            if not n.startswith("backbone."):
                yield n, p

    # ------------------------------------------------------------------ forward
    def encode(self, input_ids, bbox, attention_mask, pixel_values=None):
        kw = dict(input_ids=input_ids, bbox=bbox, attention_mask=attention_mask)
        if self.visual and pixel_values is not None:
            kw["pixel_values"] = pixel_values
        out = self.backbone(**kw)
        return out.last_hidden_state[:, : input_ids.size(1)]  # text tokens come first

    def tag_logits(self, seq):
        return self.tagger(self.drop(seq))

    def pool(self, seq_i: torch.Tensor, ent_tok: list) -> torch.Tensor:
        n = len(ent_tok)
        L = max(len(t) for t in ent_tok)
        idx = seq_i.new_zeros((n, L), dtype=torch.long)
        mask = seq_i.new_zeros((n, L))
        for i, t in enumerate(ent_tok):
            idx[i, : len(t)] = torch.as_tensor(t, device=seq_i.device)
            mask[i, : len(t)] = 1
        gathered = seq_i[idx]                                      # [n, L, H]
        pooled = (gathered * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        return self.ent(self.drop(pooled))

    def link_logits(self, seq_i, ent_tok, ent_box, keys, values, scale):
        """-> [V, K+1] logits; last column is 'no key'."""
        if not values:
            return seq_i.new_zeros((0, len(keys) + 1))
        e = self.pool(seq_i, ent_tok)
        boxes = torch.as_tensor(ent_box, dtype=e.dtype, device=e.device)
        ev = e[values]
        null = self.null(ev)                                        # [V, 1]
        if not keys:
            return null
        ek = e[keys]
        V, K = ev.size(0), ek.size(0)
        a = ek.unsqueeze(0).expand(V, K, -1)
        b = ev.unsqueeze(1).expand(V, K, -1)
        g = self.geom(geometry_features(boxes[keys].float(), boxes[values].float(), scale, self.mode).to(e.dtype))
        s = self.pair(torch.cat([a, b, a * b, g], dim=-1)).squeeze(-1)  # [V, K]
        return torch.cat([s, null], dim=1)


def link_loss(logits: torch.Tensor, targets: list) -> tuple[torch.Tensor, int]:
    """Value-to-key softmax loss with a 'no key' column. targets[i]: list of key columns,
    [] for 'no key', None for 'unknown' (gold key in another chunk -> skipped)."""
    losses = []
    null_col = logits.size(1) - 1
    logp = F.log_softmax(logits.float(), dim=1)
    for i, t in enumerate(targets):
        if t is None:
            continue
        if not t:
            losses.append(-logp[i, null_col])
        else:
            losses.append(-torch.logsumexp(logp[i, t], dim=0))
    if not losses:
        return logits.new_zeros(()), 0
    return torch.stack(losses).sum(), len(losses)


def tag_loss(tag_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Mean cross-entropy over SUPERVISED words. A batch with none (e.g. two chunks of long SROIE receipts
    that hold none of the four labelled values) gives 0, not NaN."""
    y = labels.reshape(-1)
    n = (y != -100).sum()
    total = F.cross_entropy(tag_logits.float().reshape(-1, tag_logits.size(-1)), y, ignore_index=-100, reduction="sum")
    return total / n.clamp(min=1)


def count_parameters(m: nn.Module) -> str:
    n = sum(p.numel() for p in m.parameters())
    h = sum(p.numel() for _, p in m.head_parameters()) if hasattr(m, "head_parameters") else 0
    return f"{n / 1e6:.1f}M parameters ({h / 1e6:.2f}M in heads)"

