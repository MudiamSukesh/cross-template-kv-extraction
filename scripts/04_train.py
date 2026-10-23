r"""Step 4 - train the tag + link model on TRAIN templates only.

  python scripts\04_train.py --config configs\base.yaml
  python scripts\04_train.py --config configs\base.yaml --set train.epochs=5 train.out_dir=runs\smoke

--set overrides any config key (dotted path), which is how ablation runs are made.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kvx.config import load_config  # noqa: E402
from kvx.train import train  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--set", nargs="*", default=[], help="overrides, e.g. train.epochs=5 model.visual=true")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    train(cfg)


if __name__ == "__main__":
    main()
