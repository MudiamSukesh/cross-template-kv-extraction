r"""Step 5 - seen vs unseen evaluation of a trained run.

  python scripts\05_evaluate.py --run runs\base
  python scripts\05_evaluate.py --run runs\base --test-docs cache\vrdu_doctr.jsonl   (VRDU through docTR OCR)

Writes runs\base\eval\: metrics.json, report.txt, per_template_<split>.csv, predictions.jsonl,
qualitative\ (gold on the left, prediction on the right; magenta = wrong link).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kvx.evaluate import run_evaluation  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--docs", help="override the docs file stored in the run config")
    ap.add_argument("--splits", nargs="*", help="default: val seen_test unseen_test")
    ap.add_argument("--test-docs", nargs="*", help="held-out files (default: data.test_docs of the run, i.e. VRDU)")
    ap.add_argument("--max-test-docs", type=int, help="evaluate only the first N held-out pages (quick check)")
    args = ap.parse_args()
    run_evaluation(args.run, args.docs, args.splits, args.test_docs, args.max_test_docs)


if __name__ == "__main__":
    main()
