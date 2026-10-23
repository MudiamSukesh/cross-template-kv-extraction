r"""Step 1b - build the held-out VRDU test set (one document per page, never trained on).

  $D = "C:\Users\sukes\Documents\cross-template-form-extraction\data\dataset"
  python scripts\01b_build_vrdu.py --vrdu $D\vrdu --out cache\vrdu.jsonl                 # VRDU's own OCR
  python scripts\01b_build_vrdu.py --vrdu $D\vrdu --ocr doctr --out cache\vrdu_doctr.jsonl  # our docTR OCR

--max-docs N limits documents per corpus (quick runs). Pages without any annotated value are skipped
unless --all-pages (they cannot change value recall). The coordinate order of VRDU's boxes is detected
from the data and printed.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kvx.loaders_extra import iter_vrdu  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vrdu", required=True, help="the vrdu folder holding registration-form/ and ad-buy-form/")
    ap.add_argument("--corpora", default="registration-form,ad-buy-form")
    ap.add_argument("--ocr", choices=["provided", "doctr"], default="provided")
    ap.add_argument("--ocr-cache", help="docTR results cache (default: cache/doctr_vrdu)")
    ap.add_argument("--max-docs", type=int, default=None)
    ap.add_argument("--all-pages", action="store_true")
    ap.add_argument("--bbox-order", choices=["auto", "xyxy", "xxyy"], default="auto")
    ap.add_argument("--out", default="cache/vrdu.jsonl")
    args = ap.parse_args()

    stats = Counter()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    templates, sources = set(), Counter()
    with open(out, "w", encoding="utf-8") as f:
        for doc in iter_vrdu(args.vrdu, stats, corpora=[c.strip() for c in args.corpora.split(",") if c.strip()],
                             ocr=args.ocr, max_docs=args.max_docs, all_pages=args.all_pages,
                             ocr_cache=args.ocr_cache or "cache/doctr_vrdu", bbox_order=args.bbox_order):
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
            n += 1
            templates.add(doc["template_id"])
            sources[doc["source"]] += 1
            if n % 500 == 0:
                print(f"  {n} pages ...")
    print(f"\nWrote {n} VRDU pages to {out} ({', '.join(f'{k}={v}' for k, v in sorted(sources.items()))}); "
          f"{len(templates)} template groups")
    print("\nCounters:")
    for k in sorted(stats):
        print(f"  {k:45s} {stats[k]}")


if __name__ == "__main__":
    main()
