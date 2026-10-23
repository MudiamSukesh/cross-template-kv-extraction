"""Drawing helpers: gold annotations, predictions, and audit montages."""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from .schema import centre, union_box

COLOR = {"key": (30, 100, 220), "value": (20, 150, 60), "header": (230, 130, 0), "other": (160, 160, 160)}
LINK = (220, 30, 30)
WRONG = (255, 0, 200)
HEADER_LINK = (140, 60, 200)


def page_image(doc: dict) -> Image.Image:
    """The page as an RGB image in the document's own pixel frame. VRDU pages are rendered from the PDF
    on demand (pdf_path + page_index) and resized to the OCR page size so the boxes line up."""
    p = doc.get("image_path")
    if p and Path(p).exists():
        return Image.open(p).convert("RGB")
    pdf = doc.get("pdf_path")
    if pdf and Path(pdf).exists():
        try:
            import pypdfium2 as pdfium
            d = pdfium.PdfDocument(pdf)
            page = d[int(doc.get("page_index", 0))]
            scale = max(int(doc["width"]) / max(page.get_width(), 1.0), 0.5)
            im = page.render(scale=scale).to_pil().convert("RGB")
            d.close()
            return im.resize((int(doc["width"]), int(doc["height"])))
        except Exception:  # noqa: BLE001 - drawing must never break evaluation
            pass
    return Image.new("RGB", (int(doc["width"]), int(doc["height"])), "white")


def _ent_boxes(words, entities):
    return [union_box(words[i]["box"] for i in e["word_ids"]) for e in entities]


def draw_entities_links(im, words, entities, links, header_links=(), wrong_links=(), width=2):
    d = ImageDraw.Draw(im)
    boxes = _ent_boxes(words, entities)
    for e, b in zip(entities, boxes):
        d.rectangle(b, outline=COLOR.get(e["role"], (0, 0, 0)), width=width)
    for h, k in header_links:
        d.line([centre(boxes[h]), centre(boxes[k])], fill=HEADER_LINK, width=1)
    for k, v in links:
        p, q = centre(boxes[k]), centre(boxes[v])
        d.line([p, q], fill=LINK, width=width)
        d.ellipse([q[0] - 4, q[1] - 4, q[0] + 4, q[1] + 4], fill=LINK)
    for k, v in wrong_links:
        p, q = centre(boxes[k]), centre(boxes[v])
        d.line([p, q], fill=WRONG, width=width)
        d.ellipse([q[0] - 4, q[1] - 4, q[0] + 4, q[1] + 4], fill=WRONG)
    return im


def draw_gold(doc: dict) -> Image.Image:
    im = page_image(doc)
    if not doc["entities"] and doc.get("gold_values"):  # VRDU: gold values only - green boxes, grey OCR words
        d = ImageDraw.Draw(im)
        for w in doc["words"]:
            d.rectangle(w["box"], outline=COLOR["other"], width=1)
        for g in doc["gold_values"]:
            ids = g.get("word_ids") or []
            if ids:
                b = union_box(doc["words"][i]["box"] for i in ids)
                d.rectangle(b, outline=COLOR["value"], width=3)
                d.text((b[0], max(0, b[1] - 12)), str(g.get("field") or ""), fill=COLOR["value"])
        return im
    return draw_entities_links(im, doc["words"], doc["entities"], doc["links"], doc.get("header_links", []))


def draw_prediction(doc: dict, pred: dict, gold_links=None) -> Image.Image:
    """pred = output of kvx.predict.predict_doc ("pred_entities", "pred_links" over doc["words"]).
    If gold_links (a set of (key word-ids, value word-ids) tuples) is given, wrong links are magenta."""
    im = page_image(doc)
    ents, links = pred["pred_entities"], pred["pred_links"]
    good, bad = links, []
    if gold_links is not None:
        good, bad = [], []
        for k, v in links:
            sig = (tuple(sorted(ents[k]["word_ids"])), tuple(sorted(ents[v]["word_ids"])))
            (good if sig in gold_links else bad).append((k, v))
    return draw_entities_links(im, doc["words"], ents, good, wrong_links=bad)


def montage(docs: list[dict], out_path, thumb=220, cols=6, title_fn=None):
    """Grid of page thumbnails with word boxes - used to audit template clusters by eye."""
    if not docs:
        return None
    rows = (len(docs) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * thumb, rows * (thumb + 18)), "white")
    d = ImageDraw.Draw(canvas)
    for i, doc in enumerate(docs):
        im = page_image(doc)
        dd = ImageDraw.Draw(im)
        for w in doc["words"]:
            dd.rectangle(w["box"], outline=(30, 100, 220), width=max(1, im.width // 400))
        im.thumbnail((thumb, thumb))
        x, y = (i % cols) * thumb, (i // cols) * (thumb + 18)
        canvas.paste(im, (x, y + 18))
        label = title_fn(doc) if title_fn else doc["doc_id"]
        d.text((x + 2, y + 2), label[:34], fill=(0, 0, 0))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    return out_path
