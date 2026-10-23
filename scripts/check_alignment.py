r"""Check automatically that word boxes sit on the printed text (no eyeballing needed).

  python scripts\check_alignment.py --docs cache\vrdu.jsonl
  python scripts\check_alignment.py --docs cache\docs.jsonl --n 30

For each sampled page: render it and count the word boxes that contain ink (>= 2% dark pixels). The
same is measured with every box moved down by 2.5 box heights, which puts it on the gap between lines.
Boxes on the text: almost all hit ink, the shifted copies far fewer. Boxes that are shifted, swapped
(x/y order) or scaled wrongly: both numbers are low or similar. Pages without an image are skipped.
Exit code 1 when the boxes are not aligned.
"""
import argparse
import random
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kvx.schema import iter_jsonl  # noqa: E402
from kvx.viz import page_image  # noqa: E402


def hit_rates(doc) -> tuple[float, float] | None:
    """(share of boxes with ink, share of shifted boxes with ink)"""
    im = page_image(doc)
    g = 1.0 - np.asarray(im.convert("L"), dtype=np.float32) / 255.0      # ink = 1, paper = 0
    if g.mean() < 1e-5:
        return None                                                        # blank: no page image
    H, W = g.shape
    dark = g > 0.5

    def hit(box):
        x0, y0, x1, y1 = [int(round(v)) for v in box]
        x0, x1, y0, y1 = max(0, x0), min(W, x1), max(0, y0), min(H, y1)
        if x1 <= x0 or y1 <= y0:
            return None
        return dark[y0:y1, x0:x1].mean() >= 0.02
    on, off = [], []
    for w in doc["words"]:
        b = w["box"]
        h = b[3] - b[1]
        a = hit(b)
        c = hit([b[0], b[1] + 2.5 * h, b[2], b[3] + 2.5 * h])
        if a is not None and c is not None:
            on.append(a)
            off.append(c)
    if len(on) < 3:
        return None
    return sum(on) / len(on), sum(off) / len(off)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--min-hit", type=float, default=0.6, help="median share of boxes that must contain ink")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng, sample, i = random.Random(args.seed), [], 0
    for d in iter_jsonl(args.docs):
        if len(sample) < args.n:
            sample.append(d)
        else:
            j = rng.randint(0, i)
            if j < args.n:
                sample[j] = d
        i += 1
    on_all, off_all = [], []
    for d in sample:
        r = hit_rates(d)
        tag = "skipped (no page image)" if r is None else f"on-text {r[0]:.2f}  shifted {r[1]:.2f}"
        print(f"  {tag:>30s}  {d['doc_id']}")
        if r is not None:
            on_all.append(r[0])
            off_all.append(r[1])
    if not on_all:
        sys.exit("no page could be rendered - cannot check alignment")
    on, off = statistics.median(on_all), statistics.median(off_all)
    ok = on >= args.min_hit and on >= 1.5 * off
    print(f"\nboxes containing ink: {on:.2f} (shifted control {off:.2f}) over {len(on_all)} pages "
          f"-> {'ALIGNED' if ok else 'NOT ALIGNED (boxes do not sit on the text)'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
