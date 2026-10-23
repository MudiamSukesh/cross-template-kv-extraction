r"""Step 1 - convert the TRAINING datasets into one unified JSONL file.

PowerShell example (run from the project folder):
  $D = "C:\Users\sukes\Documents\cross-template-form-extraction\data\dataset"
  python scripts\01_build_data.py --funsd $D\FUNSD --sroie $D\SROIE2019 --wildreceipt $D\wildreceipt --out cache\docs.jsonl

VRDU (the held-out test set) is built separately by scripts\01b_build_vrdu.py.
Prints every drop counter and the structural statistics you need for the dataset slide.
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kvx.loaders import dataset_stats, load_custom, load_funsd, load_labelstudio, load_xfund  # noqa: E402
from kvx.loaders_extra import load_sroie, load_wildreceipt  # noqa: E402
from kvx.schema import write_jsonl  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--funsd", help="FUNSD 'dataset' folder (contains training_data, testing_data)")
    ap.add_argument("--sroie", help="SROIE2019 folder (train/ and test/ with box, entities, img)")
    ap.add_argument("--wildreceipt", help="wildreceipt folder (class_list.txt, train.txt, test.txt, image_files)")
    ap.add_argument("--xfund", help="folder with XFUND {lang}.{split}.json files and image folders")
    ap.add_argument("--xfund-langs", default="de,es,fr,it,pt",
                    help="comma list; LayoutLMv3-base is English-only, so Latin-script languages are the default")
    ap.add_argument("--custom", help="self-curated forms root: <template>/annotations/*.json + <template>/images/")
    ap.add_argument("--custom-lang", default="en")
    ap.add_argument("--labelstudio", help="Label Studio JSON export of the self-curated forms (alternative to --custom)")
    ap.add_argument("--labelstudio-images", help="folder holding the exported images")
    ap.add_argument("--labelstudio-templates", help="CSV with columns image,template")
    ap.add_argument("--out", default="cache/docs.jsonl")
    args = ap.parse_args()
    if not (args.funsd or args.sroie or args.wildreceipt or args.xfund or args.custom or args.labelstudio):
        ap.error("give at least one of --funsd, --sroie, --wildreceipt, --xfund, --custom, --labelstudio")

    stats = Counter()
    docs = []
    if args.funsd:
        d = load_funsd(args.funsd, stats)
        print(f"FUNSD : {len(d)} documents")
        docs += d
    if args.sroie:
        d = load_sroie(args.sroie, stats)
        print(f"SROIE : {len(d)} documents in {len({x['template_id'] for x in d})} merchants")
        docs += d
    if args.wildreceipt:
        d = load_wildreceipt(args.wildreceipt, stats)
        print(f"WildReceipt: {len(d)} documents in {len({x['template_id'] for x in d})} stores")
        docs += d
    if args.xfund:
        langs = [x.strip() for x in args.xfund_langs.split(",") if x.strip()]
        d = load_xfund(args.xfund, langs, stats)
        print(f"XFUND : {len(d)} documents ({', '.join(sorted({x['lang'] for x in d}))})")
        docs += d
    if args.custom:
        d = load_custom(args.custom, stats, lang=args.custom_lang)
        print(f"custom: {len(d)} documents in {len({x['template_id'] for x in d})} templates")
        docs += d
    if args.labelstudio:
        d = load_labelstudio(args.labelstudio, args.labelstudio_images, stats, args.labelstudio_templates,
                             lang=args.custom_lang)
        print(f"Label Studio: {len(d)} documents in {len({x['template_id'] for x in d})} templates")
        docs += d

    ids = [d["doc_id"] for d in docs]
    dup = [k for k, c in Counter(ids).items() if c > 1]
    if dup:
        sys.exit(f"duplicate doc_ids: {dup[:5]}")

    n = write_jsonl(args.out, docs)
    print(f"\nWrote {n} documents to {args.out}")
    print("\nDrop / warning counters (all should be small; read any non-zero line):")
    for k in sorted(stats):
        print(f"  {k:40s} {stats[k]}")
    print("\nStructure per source:")
    for k, v in sorted(dataset_stats(docs).items()):
        print(f"  {k:40s} {v}")


if __name__ == "__main__":
    main()
