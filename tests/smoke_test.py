r"""End-to-end smoke test - run this FIRST on your machine (a few minutes, CPU or GPU).

  python tests\smoke_test.py

It writes synthetic data in the exact on-disk formats of FUNSD, SROIE2019, WildReceipt and VRDU, builds a
tiny random LayoutLMv3, and runs every step through the same scripts you use on the real data:
train on FUNSD + SROIE + WildReceipt, test on VRDU (its own OCR, and docTR OCR), failure analysis, and
the interface including a PDF upload. docTR runs WITHOUT weights here (offline); the real OCR model is
checked by scripts\00_check_env.py. The numbers printed mean nothing.
Work files go to smoke_work\ (delete it afterwards).
"""
import io
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "smoke_work"
ENV = dict(os.environ, KVX_OCR_UNTRAINED="1")


def run(args, label):
    t0 = time.time()
    print(f"--> {label}")
    r = subprocess.run([sys.executable] + [str(a) for a in args], cwd=ROOT, capture_output=True, text=True, env=ENV)
    if r.returncode != 0:
        print(r.stdout[-3000:])
        print(r.stderr[-3000:])
        sys.exit(f"FAILED at: {label}")
    print(f"    ok ({time.time() - t0:.0f}s)")
    return r.stdout


def must_exist(*paths):
    for p in paths:
        if not Path(p).exists():
            sys.exit(f"FAILED: expected output missing: {p}")


def main():
    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir()
    syn = WORK / "data"
    run(["tests/test_core.py"], "unit tests")
    run(["tests/make_synthetic.py", "--out", syn], "synthetic FUNSD / SROIE / WildReceipt / VRDU")
    out = run(["scripts/01_build_data.py", "--funsd", syn / "FUNSD/dataset", "--sroie", syn / "SROIE2019",
               "--wildreceipt", syn / "wildreceipt", "--out", WORK / "cache/docs.jsonl"], "step 1  training data")
    for s in ("FUNSD", "SROIE", "WildReceipt"):
        assert s in out, f"{s} missing from step 1 output"
    out = run(["scripts/01b_build_vrdu.py", "--vrdu", syn / "vrdu", "--out", WORK / "cache/vrdu.jsonl"],
              "step 1b VRDU test set (VRDU OCR)")
    assert "bbox_order" in out and "Wrote" in out
    out = run(["scripts/check_alignment.py", "--docs", WORK / "cache/vrdu.jsonl", "--n", "8"], "check VRDU boxes sit on the text")
    assert "ALIGNED" in out and "NOT ALIGNED" not in out
    run(["scripts/01b_build_vrdu.py", "--vrdu", syn / "vrdu", "--ocr", "doctr", "--max-docs", "2",
         "--ocr-cache", WORK / "cache/doctr_vrdu", "--out", WORK / "cache/vrdu_doctr.jsonl"],
        "step 1b VRDU test set (docTR OCR, untrained weights)")
    out = run(["scripts/02_template_split.py", "--docs", WORK / "cache/docs.jsonl", "--out", WORK / "cache/split"],
              "step 2  templates + split")
    if "PASSED" not in out:
        sys.exit("FAILED: leakage assertions did not report PASSED")
    run(["scripts/03_viz_gold.py", "--docs", WORK / "cache/docs.jsonl", "--n", "3", "--out", WORK / "viz"], "step 3  draw gold")
    run(["tests/make_tiny_backbone.py", "--out", WORK / "tiny", "--corpus", WORK / "cache/docs.jsonl"], "tiny backbone")
    run_dir = WORK / "runs/smoke"
    common = ["--config", "configs/base.yaml", "--set", f"data.docs={(WORK / 'cache/split/docs_split.jsonl').as_posix()}",
              f"data.test_docs=[{(WORK / 'cache/vrdu.jsonl').as_posix()}]",
              f"model.backbone={(WORK / 'tiny').as_posix()}", "train.lr=0.001", "train.head_lr=0.003", "train.patience=50"]
    out = run(["scripts/04_train.py"] + common + ["train.epochs=5", f"train.out_dir={run_dir.as_posix()}"],
              "step 4  train (text + layout)")
    run(["scripts/04_train.py"] + common + ["train.epochs=1", "model.visual=true",
         f"train.out_dir={(WORK / 'runs/visual').as_posix()}"], "step 4b train (with image stream)")
    out = run(["scripts/05_evaluate.py", "--run", run_dir], "step 5  evaluate (seen vs VRDU)")
    must_exist(run_dir / "eval/metrics.json", run_dir / "eval/report.txt", run_dir / "eval/predictions.jsonl")
    m = json.loads((run_dir / "eval/metrics.json").read_text())
    assert {"seen_test", "unseen_test", "val"} <= set(m["splits"]) and "value_recall" in m.get("gap", {}), "metrics incomplete"
    assert any(s.startswith("vrdu") for s in m["splits"]["unseen_test"]["per_source"]), "VRDU not evaluated"
    run(["scripts/05_evaluate.py", "--run", run_dir, "--splits", "unseen_test", "--test-docs",
         WORK / "cache/vrdu_doctr.jsonl"], "step 5b evaluate on docTR-OCR VRDU")
    run(["scripts/05_evaluate.py", "--run", run_dir], "step 5  (re-run for step 6)")
    run(["scripts/06_failure_analysis.py", "--run", run_dir, "--probe-docs", "6"], "step 6  failure analysis")
    must_exist(run_dir / "analysis/failure_report.md")
    out = run(["scripts/07_vrdu_benchmark.py", "--run", run_dir, "--vrdu", syn / "vrdu", "--bootstrap", "50"],
              "step 7  VRDU benchmark-style field scores (saved predictions)")
    must_exist(run_dir / "eval/vrdu_benchmark.json")
    vb = json.loads((run_dir / "eval/vrdu_benchmark.json").read_text())
    assert set(vb["corpora"]) == {"registration-form", "ad-buy-form"}, "both VRDU corpora scored"
    assert "line_item" in vb["corpora"]["ad-buy-form"]["all_pages"]["fields"], "line items scored"

    print("--> interface")
    os.environ["KVX_OCR_UNTRAINED"] = "1"
    sys.path.insert(0, str(ROOT / "app"))
    sys.path.insert(0, str(ROOT))
    from app import create_app  # noqa: E402
    client = create_app(run_dir).test_client()
    r = client.get("/")
    assert r.status_code == 200 and b"Held-out examples" in r.data, "interface home page"
    ex = client.get("/api/examples").get_json()["examples"]
    assert any(e["split"] == "unseen_test" for e in ex) and any(e["split"] == "seen_test" for e in ex), "examples"
    for e in (next(e for e in ex if e["split"] == "unseen_test"), next(e for e in ex if e["split"] == "seen_test")):
        j = client.post("/api/example", json={"doc_id": e["doc_id"]}).get_json()
        P = j["pages"][0]
        assert P["words"] and "gold" in P and P["gold"]["items"], f"example payload {e['doc_id']}"
        assert client.get(P["image"]).status_code == 200, "page image served"
    m = client.get("/api/metrics").get_json()
    assert m["metrics"] and "unseen_test" in m["metrics"]["splits"], "metrics endpoint"
    pdf = next((syn / "vrdu/registration-form/main/pdfs").glob("*.pdf"))
    r = client.post("/api/extract", data={"file": (io.BytesIO(pdf.read_bytes()), "form.pdf")}, content_type="multipart/form-data")
    j = r.get_json()
    assert r.status_code == 200 and len(j["pages"]) == 2, f"PDF upload should give 2 pages: {j}"
    ann = next((syn / "FUNSD/dataset/training_data/annotations").glob("*.json"))
    r = client.post("/api/extract", data={"file": (io.BytesIO(ann.read_bytes()), "form.json")}, content_type="multipart/form-data")
    assert r.status_code == 200 and r.get_json()["pages"][0]["gold"]["items"], "FUNSD json upload"
    r = client.post("/api/extract", data={"file": (io.BytesIO(b"x"), "notes.txt")}, content_type="multipart/form-data")
    assert r.status_code == 400 and "not supported" in r.get_json()["error"], "bad file type rejected"
    print("    ok")
    print("\nSMOKE TEST PASSED - the pipeline runs end to end in this environment.")
    print(f"(Work files: {WORK} - safe to delete.)")


if __name__ == "__main__":
    main()
