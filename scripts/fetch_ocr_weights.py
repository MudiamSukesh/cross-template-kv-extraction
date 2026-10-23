r"""Download and verify docTR's OCR weights, or tell you exactly where to put them.

  python scripts\fetch_ocr_weights.py                       # fast_base + parseq (default)
  python scripts\fetch_ocr_weights.py --reco crnn_vgg16_bn  # the faster recognizer

If the download fails (SSL errors from antivirus or proxy HTTPS inspection), open each printed URL in your
browser, save the file, and move it to the printed path. This script then verifies it.
Other OCR models can be chosen without code changes: $env:KVX_OCR_DET="db_resnet50"; $env:KVX_OCR_RECO="crnn_vgg16_bn"
"""
import argparse
import hashlib
import shutil
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kvx.ocr import DEFAULT_DET, DEFAULT_RECO, use_system_certificates, weight_urls  # noqa: E402


def sha_ok(path: Path) -> bool:
    prefix = path.stem.rsplit("-", 1)[-1]
    h = hashlib.sha256(path.read_bytes()).hexdigest()
    return h.startswith(prefix)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", default=DEFAULT_DET)
    ap.add_argument("--reco", default=DEFAULT_RECO)
    args = ap.parse_args()
    print("system certificate store:", "in use" if use_system_certificates() else "truststore not installed")
    cache_dir = None
    all_ok = True
    for arch, (url, path) in weight_urls(args.det, args.reco).items():
        cache_dir = path.parent
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if sha_ok(path):
                print(f"[ok]      {arch:15s} {path}")
                continue
            path.unlink()
            print(f"[removed] {arch:15s} incomplete file that failed its checksum")
        print(f"[fetch]   {arch:15s} {url}")
        try:
            tmp = path.with_suffix(".part")
            with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
                shutil.copyfileobj(r, f)
            tmp.replace(path)
            if not sha_ok(path):
                path.unlink()
                raise IOError("checksum mismatch")
            print(f"[ok]      {arch:15s} downloaded and verified")
        except Exception as e:  # noqa: BLE001
            all_ok = False
            print(f"[FAILED]  {arch:15s} {type(e).__name__}: {e}")
            print(f"          -> open this URL in a browser and save the file:\n             {url}")
            print(f"          -> move it to exactly:\n             {path}")
    if cache_dir and cache_dir.exists():
        print("\nfiles in the docTR cache:", ", ".join(sorted(p.name for p in cache_dir.glob('*.pt'))) or "(none)")
    print("\nAll OCR weights present." if all_ok else "\nRe-run this script after placing the files to verify them.")


if __name__ == "__main__":
    main()
