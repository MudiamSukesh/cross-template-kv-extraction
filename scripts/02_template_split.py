r"""Step 2 - induce templates (FUNSD/XFUND), split by template, check for leakage.

  python scripts\02_template_split.py --docs cache\docs.jsonl --out cache\split `
      --custom-unseen custom/form_a,custom/form_b

Writes into --out:
  docs_split.jsonl         documents with template_id filled and a "split" field
  split_summary.txt        counts per source and split, threshold, leakage checks
  clusters.csv             doc -> template, nearest neighbour and its distance
  nn_distance_hist.txt     histogram of nearest-neighbour distances (to pick the threshold)
  montages/<template>.png  every multi-document cluster, for the by-eye audit

Re-run with a different --threshold-<source> until the montages look right, BEFORE training.
"""
import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kvx.schema import read_jsonl, write_jsonl  # noqa: E402
from kvx.templates import check_split, induce_templates, near_duplicates, split_templates  # noqa: E402
from kvx.viz import montage  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", default="cache/docs.jsonl")
    ap.add_argument("--out", default="cache/split")
    ap.add_argument("--threshold-funsd", type=float, default=0.35)
    ap.add_argument("--threshold-xfund", type=float, default=0.25)
    ap.add_argument("--alpha-funsd", type=float, default=0.5, help="weight of layout vs key text")
    ap.add_argument("--alpha-xfund", type=float, default=1.0,
                    help="1.0 = layout only, so the same template in two languages still merges")
    ap.add_argument("--grid", type=int, default=12)
    ap.add_argument("--unseen-frac", type=float, default=0.0,
                    help="share of each TRAINING source's templates also held out as unseen (VRDU is the main "
                         "unseen test, built by 01b_build_vrdu.py); 0 = none")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seen-frac", type=float, default=0.2)
    ap.add_argument("--custom-unseen", default="", help="comma list of custom template ids held out")
    ap.add_argument("--custom-val", default="", help="comma list of custom template ids for validation")
    ap.add_argument("--dup-threshold", type=float, default=0.10)
    ap.add_argument("--no-montages", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    docs = read_jsonl(args.docs)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    by_src = defaultdict(list)
    for d in docs:
        by_src[d["source"]].append(d)

    lines, csv_rows, hist_lines, dups = [], ["doc_id,source,template_id,nn_doc,nn_distance"], [], []
    assign = {}
    induced_sources = {d["source"] for d in docs if d["template_id"] is None}
    if any(str(d["source"]).startswith("vrdu") for d in docs):
        raise SystemExit("VRDU documents must not be in the training split file - build them with 01b_build_vrdu.py")
    for src, sdocs in sorted(by_src.items()):
        sdocs.sort(key=lambda d: d["doc_id"])
        if any(d["template_id"] is None for d in sdocs):  # no template labels -> induce them
            thr = getattr(args, f"threshold_{src}", args.threshold_funsd)
            alpha = getattr(args, f"alpha_{src}", args.alpha_funsd)
            D, _, nn_info = induce_templates(sdocs, src, alpha=alpha, threshold=thr, grid=args.grid)
            nn_d = np.array([x[2] for x in nn_info if np.isfinite(x[2])])
            if len(nn_d):
                counts, edges = np.histogram(nn_d, bins=20, range=(0, max(1.0, float(nn_d.max()))))
                hist_lines.append(f"[{src}] nearest-neighbour distance (threshold={thr}, alpha={alpha})")
                for c, e0, e1 in zip(counts, edges, edges[1:]):
                    mark = "  <- threshold" if e0 <= thr < e1 else ""
                    hist_lines.append(f"  {e0:5.2f}-{e1:5.2f} {'#' * int(c):40s} {c}{mark}")
            nn_map = {a: (b, dist) for a, b, dist in nn_info}
            forced_u, forced_v = (), ()
        else:  # template ids already given: custom folders, SROIE merchant, WildReceipt store name.
            # No layout near-duplicate check here: receipts of different shops share one layout, so it
            # would only raise false alarms - the given ids are the template definition.
            D = None
            nn_map = {}
            forced_u = tuple(t for t in args.custom_unseen.split(",") if t.strip()) if src == "custom" else ()
            forced_v = tuple(t for t in args.custom_val.split(",") if t.strip()) if src == "custom" else ()
        for d in sdocs:
            nb, dist = nn_map.get(d["doc_id"], ("", float("nan")))
            csv_rows.append(f"{d['doc_id']},{src},{d['template_id']},{nb},{dist:.4f}")

        a = split_templates(sdocs, args.unseen_frac, args.val_frac, args.seen_frac, args.seed,
                            forced_unseen=forced_u, forced_val=forced_v,
                            min_unseen_templates=2 if (args.unseen_frac > 0 or forced_u) else 0)
        n_t = check_split(sdocs, a)
        if D is not None:
            dups += near_duplicates(sdocs, D, a, args.dup_threshold)
        assign.update(a)
        c = Counter(a.values())
        nt = len({d["template_id"] for d in sdocs})
        multi = sum(1 for k, v in Counter(d["template_id"] for d in sdocs).items() if v > 1)
        lines.append(f"[{src}] {len(sdocs)} docs, {nt} templates ({multi} with more than one document)")
        for s in ("train", "val", "seen_test", "unseen_test"):
            lines.append(f"    {s:12s} docs={c.get(s, 0):5d}  templates={n_t[s]}")

        if not args.no_montages and src in induced_sources:
            groups = defaultdict(list)
            for d in sdocs:
                groups[d["template_id"]].append(d)
            for tid, g in groups.items():
                if len(g) > 1:
                    montage(g[:24], out / "montages" / (tid.replace("/", "__") + ".png"),
                            title_fn=lambda x: f"{x['doc_id'].split('/')[-1]} [{assign[x['doc_id']]}]")

    check_split(docs, assign)  # global re-check across sources
    for d in docs:
        d["split"] = assign[d["doc_id"]]
    write_jsonl(out / "docs_split.jsonl", docs)
    (out / "clusters.csv").write_text("\n".join(csv_rows) + "\n", encoding="utf-8")
    (out / "nn_distance_hist.txt").write_text("\n".join(hist_lines) + "\n", encoding="utf-8")

    lines.append("")
    if not any(v == "unseen_test" for v in assign.values()):
        lines.append("unseen_test here is empty by design: the unseen-template test set is VRDU (cache/vrdu.jsonl).")
    lines.append("Leakage assertions: PASSED (no template in more than one of train / val / unseen_test)")
    if dups:
        lines.append(f"WARNING: {len(dups)} held-out docs are closer than {args.dup_threshold} to a train doc:")
        for a_, b_, dist in dups[:30]:
            lines.append(f"    {a_}  ~  {b_}   d={dist:.3f}")
        lines.append("  Inspect them; if they share a template, raise the threshold or force them together.")
    else:
        lines.append(f"Near-duplicate check: no held-out doc within {args.dup_threshold} of a train doc")
    txt = "\n".join(lines)
    (out / "split_summary.txt").write_text(txt + "\n", encoding="utf-8")
    print(txt)
    print(f"\nWrote {out / 'docs_split.jsonl'}; audit {out / 'montages'} and {out / 'nn_distance_hist.txt'}")


if __name__ == "__main__":
    main()
