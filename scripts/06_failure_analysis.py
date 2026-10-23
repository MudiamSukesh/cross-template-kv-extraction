r"""Step 6 - failure analysis on unseen templates (needs step 5 first).

  python scripts\06_failure_analysis.py --run runs\base

Writes runs\base\analysis\: failure_report.md, failure_analysis.json, errors\ (gold links red,
wrong predicted links magenta).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kvx.analysis import run_analysis  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--docs")
    ap.add_argument("--probe-docs", type=int, default=60, help="documents per split used for probes")
    args = ap.parse_args()
    run_analysis(args.run, args.docs, args.probe_docs)


if __name__ == "__main__":
    main()
