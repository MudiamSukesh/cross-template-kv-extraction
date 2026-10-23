r"""Interface: drop a scanned form (image or PDF) -> docTR reads words and boxes -> the model tags every word
(key / value / header / other) and links each value to its key -> boxes are drawn over the page.
Hovering a pair lights up its key, its value and the link between them.

  python app\app.py --run runs\base          then open http://127.0.0.1:5000

Tabs
  Extract      upload any form (png/jpg/tif/pdf, every page)
  Held-out     documents the model never trained on (VRDU pages, unseen documents of seen templates),
               with the gold values marked found / missed
  Results      the measured seen vs unseen numbers from scripts\05_evaluate.py

WHY A LOCAL SERVER: docTR and LayoutLMv3 both run on the GPU; the page is a thin client.
Models load ONCE at startup. Requests are serialised with a lock (and the server is single-threaded):
two inferences in one CUDA context on an 8 GB card is how you get an out-of-memory crash that looks random.

WHAT THE NUMBERS HERE ARE NOT: uploads go through docTR OCR, while the reported VRDU numbers use VRDU's own
OCR (unless you evaluated cache\vrdu_doctr.jsonl). A single upload is a demonstration, not a measurement.
"""
import argparse
import json
import sys
import tempfile
import threading
import time
import uuid
from collections import Counter
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_from_directory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kvx.loaders import _size_from_words, convert_form  # noqa: E402
from kvx.metrics import (HEADLINE, doc_counts, norm_text,  # noqa: E402
                         predicted_value_candidates, value_match)
from kvx.ocr import ocr_available, ocr_document  # noqa: E402
from kvx.predict import predict_doc  # noqa: E402
from kvx.runtime import load_run  # noqa: E402
from kvx.schema import TAGS, read_jsonl, union_box, validate_doc  # noqa: E402
from kvx.viz import page_image  # noqa: E402

UPLOADS = Path(__file__).resolve().parent / "_uploads"
IMG_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
LOCK = threading.Lock()
STATE: dict = {"max_pages": 10}
app = Flask(__name__, template_folder=str(Path(__file__).parent / "templates"))
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024
app.json.sort_keys = False  # keep metric / split / field order as computed


# ------------------------------------------------------------------ helpers
def _role(tag_id: int) -> str:
    t = TAGS[tag_id]
    return "other" if t == "O" else t.split("-", 1)[1].lower()


def _save_page(doc) -> str:
    """Page image as a file the browser can load (JPEG keeps big scans light)."""
    im = page_image(doc)
    name = f"{uuid.uuid4().hex}.jpg"
    im.save(UPLOADS / name, quality=88)
    return f"/files/{name}"


def _text(doc, ids):
    return " ".join(doc["words"][i]["text"] for i in ids)


def _box(doc, ids):
    return [round(v, 1) for v in union_box(doc["words"][i]["box"] for i in ids)]


def page_payload(doc: dict, pred: dict, image_url: str, timing: dict) -> dict:
    words = [{"t": w["text"], "b": [round(v, 1) for v in w["box"]],
              "r": _role(pred["word_tags"][i]), "c": pred.get("word_conf", [0] * len(doc["words"]))[i]}
             for i, w in enumerate(doc["words"])]
    ents = pred["pred_entities"]
    pairs, linked = [], set()
    for (k, v), p in zip(pred["pred_links"], pred.get("pred_link_p", [None] * len(pred["pred_links"]))):
        pairs.append({"key": _text(doc, ents[k]["word_ids"]), "value": _text(doc, ents[v]["word_ids"]),
                      "kb": _box(doc, ents[k]["word_ids"]), "vb": _box(doc, ents[v]["word_ids"]), "p": p})
        linked.add(v)
    for i, e in enumerate(ents):
        if e["role"] == "value" and i not in linked:
            pairs.append({"key": None, "value": _text(doc, e["word_ids"]), "kb": None,
                          "vb": _box(doc, e["word_ids"]), "p": None})
    return {"doc_id": doc["doc_id"], "image": image_url, "width": doc["width"], "height": doc["height"],
            "words": words, "counts": dict(Counter(w["r"] for w in words)), "pairs": pairs,
            "timing": timing, "n_chunks": pred["n_chunks"]}


def gold_payload(doc: dict, pred: dict) -> dict:
    """Gold values of a held-out document, each marked found / missed (same matching as the metrics)."""
    cands = predicted_value_candidates(doc, pred)
    key_of = {}
    for k, v in doc.get("links", []):
        key_of[v] = _text(doc, doc["entities"][k]["word_ids"])
    items = []
    if doc.get("gold_values") is not None:
        golds = [(g.get("field"), g["text"], None) for g in doc["gold_values"]]
    else:
        golds = [(None, _text(doc, e["word_ids"]), key_of.get(e["id"])) for e in doc["entities"] if e["role"] == "value"]
    for field, text, printed_key in golds:
        gt = norm_text(text)
        if not gt:
            continue
        items.append({"field": field, "key": printed_key, "text": text.strip(),
                      "found": any(value_match(t, gt, STATE["fuzzy"]) for t, _ in cands)})
    c = doc_counts(doc, pred, STATE["fuzzy"])
    scores = {"value recall": c["value_rec"][0] / c["value_rec"][2] if c["value_rec"][2] else None}
    if c["kv_field"][2]:
        scores["key-matched recall"] = c["kv_field"][0] / c["kv_field"][2]
    if "link" in c:
        from kvx.metrics import f1
        ek = [c["entity_key"][i] + c["entity_value"][i] for i in range(3)]
        scores["entity F1"] = f1(*ek)[0]
        scores["link F1 (gold spans)"] = f1(*c["link"])[0]
        scores["pair F1 (end-to-end)"] = f1(*c["e2e_pair"])[0]
    return {"items": items, "scores": scores, "split": doc.get("split"), "source": doc["source"],
            "template": doc.get("template_id"), "annotation": doc.get("annotation", "full")}


def _predict(doc):
    model, enc, adt = STATE["model"], STATE["encoder"], STATE["amp"]
    return predict_doc(model, enc, doc, next(model.parameters()).device, adt)


# ------------------------------------------------------------------- routes
@app.get("/")
def index():
    return render_template("index.html", state={k: STATE[k] for k in (
        "run", "epoch", "backbone", "visual", "device", "ocr", "ocr_ok", "select_metric", "val_score")})


@app.get("/files/<name>")
def files(name):
    return send_from_directory(UPLOADS, name)


@app.post("/api/extract")
def api_extract():
    f = request.files.get("file")
    if not f or not f.filename:
        return jsonify(error="no file was uploaded"), 400
    suffix = Path(f.filename).suffix.lower()
    if suffix not in IMG_EXT | {".pdf", ".json"}:
        return jsonify(error=f"{suffix or 'this file'} is not supported - use png, jpg, tif, bmp, webp or pdf"), 400
    with LOCK:
        tmp = Path(tempfile.mkdtemp(prefix="kvx_", dir=UPLOADS))
        path = tmp / f"upload{suffix}"
        f.save(path)
        try:
            if suffix == ".json":  # a FUNSD-format annotation: words come from the file, no OCR
                data = json.loads(path.read_text(encoding="utf-8"))
                if "form" not in data:
                    return jsonify(error="JSON must be FUNSD format: {\"form\": [...]}"), 400
                w, h = _size_from_words(data["form"])
                words, ents, links, hl = convert_form(data["form"], w, h, Counter())
                doc = {"doc_id": Path(f.filename).stem, "source": "upload", "template_id": None, "image_path": None,
                       "width": w, "height": h, "words": words, "entities": ents,
                       "links": [list(x) for x in links], "header_links": [list(x) for x in hl], "annotation": "full"}
                errs = validate_doc(doc)
                if errs or not words:
                    return jsonify(error="annotation problems: " + "; ".join(errs[:3] or ["no words"])), 400
                t0 = time.time()
                pred = _predict(doc)
                out = page_payload(doc, pred, _save_page(doc), {"ocr": 0.0, "model": round(time.time() - t0, 2)})
                out["gold"] = gold_payload(doc, pred)
                return jsonify(pages=[out], ocr=None)
            ok, msg = ocr_available()
            if not ok:
                return jsonify(error=f"OCR unavailable: {msg}"), 500
            t0 = time.time()
            pages = ocr_document(path, doc_id=Path(f.filename).stem, work_dir=tmp, max_pages=STATE["max_pages"])
            t_ocr = (time.time() - t0) / max(1, len(pages))
            out = []
            for d in pages:
                if not d["words"]:
                    out.append({"doc_id": d["doc_id"], "empty": True, "image": _save_page(d),
                                "width": d["width"], "height": d["height"]})
                    continue
                t1 = time.time()
                pred = _predict(d)
                out.append(page_payload(d, pred, _save_page(d),
                                        {"ocr": round(t_ocr, 2), "model": round(time.time() - t1, 2)}))
            return jsonify(pages=out, ocr=ocr_available()[1])
        except Exception as e:  # noqa: BLE001 - show the problem instead of a blank error page
            return jsonify(error=f"{type(e).__name__}: {e}"), 500


@app.get("/api/examples")
def api_examples():
    return jsonify(examples=STATE["examples"])


@app.post("/api/example")
def api_example():
    doc = STATE["docs"].get((request.get_json(silent=True) or {}).get("doc_id", ""))
    if doc is None:
        return jsonify(error="unknown document"), 404
    with LOCK:
        t0 = time.time()
        pred = _predict(doc)
        out = page_payload(doc, pred, _save_page(doc), {"ocr": 0.0, "model": round(time.time() - t0, 2)})
        out["gold"] = gold_payload(doc, pred)
    return jsonify(pages=[out], ocr=None)


@app.get("/api/metrics")
def api_metrics():
    p = Path(STATE["run"]) / "eval" / "metrics.json"
    if not p.exists():
        return jsonify(metrics=None, headline=HEADLINE)
    return jsonify(metrics=json.loads(p.read_text(encoding="utf-8")), headline=HEADLINE)


# --------------------------------------------------------------------- setup
def create_app(run_dir, docs_path=None, n_test_examples: int = 60):
    from kvx.analysis import sample_test_docs
    UPLOADS.mkdir(exist_ok=True)
    model, encoder, cfg, adt, ck = load_run(run_dir)
    ok, msg = ocr_available()
    val = ck.get("val") or {}
    STATE.update(run=str(run_dir), model=model, encoder=encoder, amp=adt, epoch=ck.get("epoch"),
                 backbone=cfg["model"]["backbone"], visual=cfg["model"]["visual"],
                 device=str(next(model.parameters()).device), ocr=msg, ocr_ok=ok,
                 fuzzy=cfg["eval"]["fuzzy_threshold"], select_metric=cfg["train"]["select_metric"],
                 val_score=val.get(cfg["train"]["select_metric"]))
    docs = {}
    dp = docs_path or cfg["data"]["docs"]
    if dp and Path(dp).exists():
        docs = {d["doc_id"]: d for d in read_jsonl(dp) if d.get("split") in ("seen_test", "unseen_test")}
    test_files = [f for f in cfg["data"].get("test_docs", []) if Path(f).exists()]
    for d in (sample_test_docs(test_files, n_test_examples) if test_files else []):
        docs[d["doc_id"]] = d
    STATE["docs"] = docs
    STATE["examples"] = sorted(({"doc_id": k, "split": v.get("split"), "source": v["source"],
                                 "template": v.get("template_id")} for k, v in docs.items()),
                               key=lambda x: (x["split"] != "unseen_test", x["source"], x["doc_id"]))
    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--docs", help="split file for held-out examples (default: from the run config)")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--max-pages", type=int, default=10, help="pages of an uploaded PDF to process")
    args = ap.parse_args()
    STATE["max_pages"] = args.max_pages
    print(f"python: {sys.executable}")
    print("loading model and examples ...")
    create_app(args.run, args.docs)
    print(f"ready: {STATE['run']} (epoch {STATE['epoch']}) on {STATE['device']}")
    print(f"OCR:   {STATE['ocr']}")
    if not STATE["ocr_ok"]:
        print("\n  !! uploads will fail until OCR is available - see the message above !!")
    print(f"\n  open  http://{args.host}:{args.port}\n")
    app.run(host=args.host, port=args.port, debug=False, threaded=False)


if __name__ == "__main__":
    main()
