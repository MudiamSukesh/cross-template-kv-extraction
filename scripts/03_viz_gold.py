r"""Draw gold entities and links on page images (sanity check after step 1 or 2).

  python scripts\03_viz_gold.py --docs cache\docs.jsonl --n 6 --out viz\gold
  python scripts\03_viz_gold.py --docs cache\docs.jsonl --doc-id funsd/82092117

Blue = key, green = value, orange = header, grey = other.
Red line = key -> value (dot on the value end). Purple = header -> key.
"""
import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kvx.schema import read_jsonl  # noqa: E402
from kvx.viz import draw_gold  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", default="cache/docs.jsonl")
    ap.add_argument("--out", default="viz/gold")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--doc-id")
    ap.add_argument("--source")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    docs = read_jsonl(args.docs)
    if args.source:
        docs = [d for d in docs if d["source"] == args.source]
    if args.doc_id:
        docs = [d for d in docs if d["doc_id"] == args.doc_id]
        if not docs:
            sys.exit(f"{args.doc_id} not found")
    else:
        random.Random(args.seed).shuffle(docs)
        docs = docs[: args.n]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for d in docs:
        p = out / (d["doc_id"].replace("/", "__") + ".png")
        draw_gold(d).save(p)
        print("saved", p)


if __name__ == "__main__":
    main()
