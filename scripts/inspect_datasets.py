r"""Print the on-disk structure of SROIE, WildReceipt and VRDU (read-only, prints a short skeleton).

  python scripts\inspect_datasets.py --root C:\Users\sukes\Documents\cross-template-form-extraction\data\dataset > inspect.txt

Send inspect.txt back. Long lists and strings are truncated, so the output stays small.
"""
import argparse
import gzip
import json
from collections import Counter
from pathlib import Path


def skel(x, depth=0, max_depth=7, max_list=2, max_str=60):
    """Compact structural skeleton of a JSON value."""
    if depth > max_depth:
        return "..."
    if isinstance(x, dict):
        return {k: skel(v, depth + 1, max_depth, max_list, max_str) for k, v in list(x.items())[:25]}
    if isinstance(x, list):
        out = [skel(v, depth + 1, max_depth, max_list, max_str) for v in x[:max_list]]
        if len(x) > max_list:
            out.append(f"... ({len(x)} items)")
        return out
    if isinstance(x, str):
        return x if len(x) <= max_str else x[:max_str] + f"...({len(x)} chars)"
    return x


def tree(root: Path, depth=3, per_dir=8):
    print(f"\n[tree] {root}")
    if not root.exists():
        print("  (missing)")
        return
    def walk(d, lvl):
        if lvl > depth:
            return
        entries = sorted(d.iterdir())
        dirs = [e for e in entries if e.is_dir()]
        files = [e for e in entries if e.is_file()]
        ext = Counter(f.suffix.lower() for f in files)
        print("  " * lvl + f"{d.name}/  ({len(dirs)} dirs, {len(files)} files: {dict(ext)})")
        for f in files[:4]:
            print("  " * (lvl + 1) + f"- {f.name}  ({f.stat().st_size} bytes)")
        for sub in dirs[:per_dir]:
            walk(sub, lvl + 1)
        if len(dirs) > per_dir:
            print("  " * (lvl + 1) + f"... {len(dirs) - per_dir} more dirs")
    walk(root, 0)


def head_text(p: Path, n=4, width=160):
    print(f"\n[head] {p}")
    try:
        with open(p, encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f):
                if i >= n:
                    break
                print("  " + line.rstrip("\n")[:width])
    except Exception as e:  # noqa: BLE001
        print(f"  (unreadable: {e})")


def first_json_record(p: Path):
    opener = gzip.open if p.suffix == ".gz" else open
    with opener(p, "rt", encoding="utf-8", errors="replace") as f:
        first = f.readline().strip()
        if first.startswith("[") or first.startswith("{") and not first.endswith("}"):
            f.seek(0)
            data = json.load(f)
            return data[0] if isinstance(data, list) else data
        return json.loads(first)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="the data\\dataset folder")
    args = ap.parse_args()
    root = Path(args.root)

    # ---------------- SROIE
    s = root / "SROIE2019"
    tree(s, depth=2)
    for sub in ("train", "test"):
        d = s / sub
        if d.exists():
            for kid in sorted(p for p in d.iterdir() if p.is_dir()):
                files = sorted(kid.glob("*"))
                txt = [f for f in files if f.suffix.lower() in (".txt", ".json", ".csv")]
                if txt:
                    head_text(txt[0], n=5)

    # ---------------- WildReceipt
    w = root / "wildreceipt"
    tree(w, depth=1)
    for name in ("class_list.txt", "dict.txt"):
        if (w / name).exists():
            head_text(w / name, n=30, width=80)
    for p in sorted(w.glob("*.txt")) + sorted(w.glob("*.json")):
        if p.name in ("class_list.txt", "dict.txt"):
            continue
        try:
            rec = first_json_record(p)
            print(f"\n[record] {p.name}  (lines: {sum(1 for _ in open(p, encoding='utf-8', errors='replace'))})")
            print(json.dumps(skel(rec), indent=1)[:2500])
        except Exception as e:  # noqa: BLE001
            head_text(p, n=2)
            print(f"  (not json lines: {e})")

    # ---------------- VRDU
    v = root / "vrdu"
    tree(v, depth=3)
    for corpus in ("registration-form", "ad-buy-form"):
        c = v / corpus
        if not c.exists():
            continue
        for p in sorted(c.rglob("meta.json"))[:1]:
            print(f"\n[meta] {p}")
            print(json.dumps(skel(json.loads(p.read_text(encoding='utf-8'))), indent=1)[:1500])
        for p in (sorted(c.rglob("dataset.jsonl*")) + sorted(c.rglob("*.jsonl")))[:1]:
            rec = first_json_record(p)
            print(f"\n[record] {p}")
            print(json.dumps(skel(rec, max_depth=14), indent=1)[:9000])
        for p in sorted(c.rglob("*split*"))[:2]:
            if p.is_file():
                print(f"\n[split file] {p}")
                try:
                    print(json.dumps(skel(json.loads(p.read_text(encoding='utf-8'))), indent=1)[:800])
                except Exception:  # noqa: BLE001
                    head_text(p, n=3)
        pdfs = sorted(c.rglob("*.pdf"))
        print(f"\n[pdfs] {corpus}: {len(pdfs)} pdf files, e.g. {[p.name for p in pdfs[:3]]}")
    print("\n(done)")


if __name__ == "__main__":
    main()
