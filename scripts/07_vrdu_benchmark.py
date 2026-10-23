r"""Step 7 - VRDU benchmark-style field extraction scores (precision / recall / F1), from SAVED predictions.

Needs step 5 first (runs\<run>\eval\predictions.jsonl). No model is run: this only re-scores the saved predictions
with VRDU's own type-specific matching (meta.json) and VRDU's precision / recall / F1 definition.

  $D = "C:\Users\sukes\Documents\cross-template-form-extraction\data\dataset"
  python scripts\07_vrdu_benchmark.py --run runs\base --vrdu $D\vrdu

  # only the documents of VRDU's own unseen-template test lists (few_shot-splits\*lv3*):
  python scripts\07_vrdu_benchmark.py --run runs\base --vrdu $D\vrdu `
      --split-files (Get-ChildItem $D\vrdu\*\few_shot-splits\*lv3*).FullName

Writes runs\<run>\eval\vrdu_benchmark.json and vrdu_benchmark.txt.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kvx.vrdu_eval import run  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--vrdu", required=True, help="the vrdu folder holding registration-form/ and ad-buy-form/")
    ap.add_argument("--test-docs", nargs="*", help="the VRDU page file that step 5 evaluated (default: from the run config)")
    ap.add_argument("--split-files", nargs="*", default=[], help="VRDU split json files; only their 'test' documents are scored")
    ap.add_argument("--corpora", default="registration-form,ad-buy-form")
    ap.add_argument("--aliases", help="alternative alias yaml (default kvx/field_aliases.yaml)")
    ap.add_argument("--bootstrap", type=int, default=1000)
    args = ap.parse_args()
    run(args.run, args.vrdu, args.test_docs, args.split_files,
        corpora=[c.strip() for c in args.corpora.split(",") if c.strip()], aliases_path=args.aliases, B=args.bootstrap)


if __name__ == "__main__":
    main()
