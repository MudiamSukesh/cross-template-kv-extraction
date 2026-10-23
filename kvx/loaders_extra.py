"""Loaders for SROIE, WildReceipt (training) and VRDU (held-out test), matched to the on-disk
formats shown by scripts/inspect_datasets.py.

SROIE2019/{train,test}/box/<id>.txt       x1,y1,x2,y2,x3,y3,x4,y4,TEXT   (one text line per row)
                       entities/<id>.txt  {"company","date","address","total"}
                       img/<id>.jpg
  -> annotation "values": the four field values are found among the words by string matching and
     tagged VALUE; every other word is UNSUPERVISED (ignore_words), because SROIE does not mark keys
     ("TOTAL" is a key but labelled nothing). No link supervision. template = merchant (company).

wildreceipt/class_list.txt, train.txt, test.txt (one JSON per line), image_files/...
  -> annotation "full": <X>_key -> KEY, <X>_value -> VALUE, Others -> O, Ignore -> unsupervised.
     Links are derived: each X_value links to the X_key of the same category on the receipt
     (nearest one if several; none if the receipt prints no such key). template = store name.

vrdu/<corpus>/main/dataset.jsonl(.gz), meta.json, pdfs/
  -> TEST ONLY. One document per page. Words and lines come from the OCR shipped with VRDU
     (default) or from docTR run on the PDF (ocr="doctr"). Gold = the annotated field values
     (annotation "values"); VRDU marks no printed keys, so no gold links exist.
"""
from __future__ import annotations

import gzip
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

from .schema import clip_box, edge_gap, median_word_height, reading_order_key, union_box, validate_doc


# ------------------------------------------------------------------------- helpers
def alnum(s: str) -> str:
    return re.sub(r"[^0-9a-z]", "", s.lower())


def split_segment(text: str, box) -> list[dict]:
    """One OCR segment (a line or line piece) -> word boxes, splitting the box by character counts."""
    toks = text.split()
    if not toks:
        return []
    x0, y0, x1, y1 = box
    total = sum(len(t) for t in toks) + len(toks) - 1
    unit = (x1 - x0) / total if total else 0.0
    out, cx = [], x0
    for t in toks:
        out.append({"text": t, "box": [cx, y0, cx + unit * len(t), y1]})
        cx += unit * (len(t) + 1)
    return out


def quad_to_box(q) -> list[float]:
    xs, ys = q[0::2], q[1::2]
    return [min(xs), min(ys), max(xs), max(ys)]


def slug(s: str, n: int = 60) -> str:
    return re.sub(r"[^0-9a-z]+", "_", (s or "").lower()).strip("_")[:n] or "unknown"


def find_value_spans(word_texts: list[str], target: str, used: set, all_occurrences: bool, stats: Counter,
                     tag: str) -> list[list[int]]:
    """Find word spans whose concatenated alphanumeric text equals the target. The first word may carry a
    short alphabetic prefix (currency "RM9.00" for "9.00"). Falls back to a fuzzy window for long values."""
    from .metrics import similarity
    t = alnum(target)
    if not t:
        stats[f"{tag}_empty_value"] += 1
        return []
    A = [alnum(w) for w in word_texts]
    spans, taken = [], set()
    for i in range(len(A)):
        if i in used or i in taken or not A[i]:
            continue
        acc = ""
        for j in range(i, len(A)):
            if j in used or j in taken:
                break
            acc += A[j]
            pre = len(acc) - len(t)
            if pre > 3:
                break
            if pre >= 0 and acc.endswith(t) and (pre == 0 or (pre < len(A[i]) and acc[:pre].isalpha())):
                spans.append(list(range(i, j + 1)))
                taken.update(range(i, j + 1))
                break
        if spans and not all_occurrences:
            break
    if spans:
        stats[f"{tag}_matched_exact"] += 1
        return spans if all_occurrences else spans[:1]
    if len(t) >= 8:  # fuzzy window for long values (company, address)
        best, best_span = 0.0, None
        for i in range(len(A)):
            acc = ""
            for j in range(i, len(A)):
                if j in used:
                    break
                acc += A[j]
                if len(acc) > 1.3 * len(t):
                    break
                if len(acc) >= 0.7 * len(t):
                    s = similarity(acc, t)
                    if s > best:
                        best, best_span = s, list(range(i, j + 1))
        if best >= 0.85:
            stats[f"{tag}_matched_fuzzy"] += 1
            return [best_span]
    stats[f"{tag}_unmatched"] += 1
    return []


def sort_lines(doc: dict) -> dict:
    """Put annotated text segments in reading order (top-to-bottom bands, then left-to-right).
    Dataset files list segments in annotation order, which is not the order an OCR engine reads."""
    if doc.get("lines"):
        tol = median_word_height(doc)
        doc["lines"].sort(key=lambda l: reading_order_key(union_box(doc["words"][i]["box"] for i in l), tol))
    return doc


def _finish(doc: dict, stats: Counter):
    errs = validate_doc(doc)
    if errs:
        raise ValueError(f"{doc['doc_id']}: invalid after conversion: {errs[:3]}")
    return doc


def _image_size(p):
    if p is not None and Path(p).exists():
        with Image.open(p) as im:
            return im.size
    return None


# --------------------------------------------------------------------------- SROIE
SROIE_FIELDS = ("company", "date", "address", "total")


def load_sroie(root, stats: Counter) -> list[dict]:
    root = Path(root)
    splits = [d for d in ("train", "test") if (root / d / "box").is_dir()]
    if not splits:
        raise FileNotFoundError(f"SROIE: expected {root}\\train\\box and {root}\\test\\box")
    docs = []
    for split in splits:
        base = root / split
        for bf in sorted((base / "box").glob("*.txt")):
            ef = base / "entities" / bf.name
            if not ef.exists():
                stats["sroie_missing_entities_file"] += 1
                continue
            try:
                ent = json.loads(ef.read_text(encoding="utf-8", errors="replace"))
            except json.JSONDecodeError:
                stats["sroie_bad_entities_json"] += 1
                continue
            img = None
            for ext in (".jpg", ".jpeg", ".png"):
                if (base / "img" / (bf.stem + ext)).exists():
                    img = base / "img" / (bf.stem + ext)
                    break
            words, lines = [], []
            for raw in bf.read_text(encoding="utf-8", errors="replace").splitlines():
                parts = raw.strip().split(",", 8)
                if len(parts) < 9:
                    if raw.strip():
                        stats["sroie_bad_box_line"] += 1
                    continue
                try:
                    q = [float(v) for v in parts[:8]]
                except ValueError:
                    stats["sroie_bad_box_line"] += 1
                    continue
                ws = split_segment(parts[8], quad_to_box(q))
                if ws:
                    lines.append(list(range(len(words), len(words) + len(ws))))
                    words += ws
            if not words:
                stats["sroie_doc_without_words"] += 1
                continue
            size = _image_size(img)
            if size is None:
                stats["sroie_image_missing"] += 1
                size = (int(max(w["box"][2] for w in words)) + 1, int(max(w["box"][3] for w in words)) + 1)
            W, H = size
            for w in words:
                w["box"] = clip_box(w["box"], W, H)
            used, entities, gold_values = set(), [], []
            texts = [w["text"] for w in words]
            for field in SROIE_FIELDS:
                val = (ent.get(field) or "").strip()
                if not val:
                    stats[f"sroie_{field}_missing"] += 1
                    continue
                gold_values.append({"field": field, "text": val})
                for span in find_value_spans(texts, val, used, all_occurrences=(field == "total"),
                                             stats=stats, tag=f"sroie_{field}"):
                    used.update(span)
                    entities.append({"id": len(entities), "role": "value", "word_ids": span})
            ignore = [i for i in range(len(words)) if i not in used]
            company = ent.get("company") or bf.stem
            docs.append(_finish(sort_lines({
                "doc_id": f"sroie/{bf.stem}", "source": "sroie", "split_hint": split, "lang": "en",
                "template_id": f"sroie/{slug(company)}", "image_path": str(img.resolve()) if img else None,
                "width": int(W), "height": int(H), "words": words, "lines": lines, "entities": entities,
                "links": [], "header_links": [], "annotation": "values", "link_supervision": False,
                "ignore_words": ignore, "gold_values": gold_values}), stats))
    return docs


# --------------------------------------------------------------------- WildReceipt
def read_class_list(path: Path) -> dict:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) == 2 and parts[0].isdigit():
            out[int(parts[0])] = parts[1].strip()
    return out


def load_wildreceipt(root, stats: Counter, files=("train.txt", "test.txt")) -> list[dict]:
    root = Path(root)
    classes = read_class_list(root / "class_list.txt")
    if not classes:
        raise FileNotFoundError(f"WildReceipt: {root / 'class_list.txt'} missing or empty")
    docs = []
    for fn in files:
        p = root / fn
        if not p.exists():
            stats[f"wildreceipt_missing_{fn}"] += 1
            continue
        split = Path(fn).stem
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            img = root / rec["file_name"]
            W, H = int(rec["width"]), int(rec["height"])
            words, lines, entities, ignore, cats = [], [], [], [], []
            store = []
            for ann in rec.get("annotations", []):
                name = classes.get(int(ann.get("label", -1)), "")
                ws = split_segment((ann.get("text") or "").strip(), quad_to_box(ann["box"]))
                if not ws:
                    stats["wildreceipt_empty_segment"] += 1
                    continue
                for w in ws:
                    w["box"] = clip_box(w["box"], W, H)
                ids = list(range(len(words), len(words) + len(ws)))
                words += ws
                lines.append(ids)
                if name == "Ignore":
                    ignore += ids
                    stats["wildreceipt_ignore_segments"] += 1
                    continue
                if name.endswith("_key"):
                    role, cat = "key", name[:-4]
                elif name.endswith("_value"):
                    role, cat = "value", name[:-6]
                else:
                    role, cat = "other", None
                if name == "Store_name_value":
                    store.append(ann.get("text") or "")
                entities.append({"id": len(entities), "role": role, "word_ids": ids})
                cats.append(cat)
            if not words:
                stats["wildreceipt_doc_without_words"] += 1
                continue
            links = []
            keys_of = defaultdict(list)
            for e, c in zip(entities, cats):
                if e["role"] == "key":
                    keys_of[c].append(e["id"])
            for e, c in zip(entities, cats):
                if e["role"] != "value":
                    continue
                ks = keys_of.get(c, [])
                if not ks:
                    stats["wildreceipt_value_without_key"] += 1
                    continue
                vb = union_box(words[w]["box"] for w in e["word_ids"])
                k = min(ks, key=lambda kid: edge_gap(union_box(words[w]["box"] for w in entities[kid]["word_ids"]), vb))
                links.append([k, e["id"]])
                if len(ks) > 1:
                    stats["wildreceipt_value_with_several_keys_nearest_used"] += 1
            stem = Path(rec["file_name"]).stem
            tmpl = f"wildreceipt/{slug(' '.join(store))}" if store else f"wildreceipt/doc_{slug(stem, 40)}"
            gold_values = [{"field": cats[e["id"]], "text": " ".join(words[w]["text"] for w in e["word_ids"])}
                           for e in entities if e["role"] == "value"]
            docs.append(_finish(sort_lines({
                "doc_id": f"wildreceipt/{slug(rec['file_name'], 120)}", "source": "wildreceipt", "split_hint": split,
                "lang": "en", "template_id": tmpl, "image_path": str(img.resolve()) if img.exists() else None,
                "width": W, "height": H, "words": words, "lines": lines, "entities": entities,
                "links": sorted(links), "header_links": [], "annotation": "full", "ignore_words": ignore,
                "gold_values": gold_values}), stats))
            if not img.exists():
                stats["wildreceipt_image_missing"] += 1
    return docs


# ----------------------------------------------------------------------------- VRDU
VRDU_CORPORA = {"registration-form": "fara", "ad-buy-form": "deepform"}


def _open_jsonl(main: Path):
    for name in ("dataset.jsonl", "dataset.jsonl.gz"):
        p = main / name
        if p.exists():
            return (gzip.open(p, "rt", encoding="utf-8") if name.endswith(".gz") else open(p, encoding="utf-8")), p
    raise FileNotFoundError(f"VRDU: no dataset.jsonl(.gz) in {main}")


def _leaf_values(name, items, out):
    """VRDU annotation values: [text, bbox, segments] leaves, possibly nested under a parent entity."""
    for it in items or []:
        if isinstance(it, list) and len(it) >= 2 and isinstance(it[0], str) and isinstance(it[1], list) \
                and it[1] and isinstance(it[1][0], (int, float)):
            out.append((name, it[0], it[1], it[2] if len(it) > 2 else []))
        elif isinstance(it, list) and len(it) == 2 and isinstance(it[0], str) and isinstance(it[1], list):
            _leaf_values(it[0], it[1], out)          # nested child entity [child_name, [...]]
        elif isinstance(it, list):
            _leaf_values(name, it, out)


def parse_vrdu_annotations(anns) -> list[tuple]:
    """-> [(field, text, bbox, segments)]. Handles simple entities ["name", [leaf, leaf, ...]] and VRDU's nested
    line items, whose name is a LIST of sub-field names: [["channel", "sub_amount"], [[leaf_channel, leaf_amount]]]
    (format defined by google-research/vrdu benchmark_utils.convert_list_to_tuple_for_entity_items)."""
    out = []
    for ent in anns or []:
        if not (isinstance(ent, list) and len(ent) == 2):
            continue
        if isinstance(ent[0], str):
            _leaf_values(ent[0], ent[1], out)
        elif isinstance(ent[0], list) and all(isinstance(n, str) for n in ent[0]) and ent[1]:
            for name, leaf in zip(ent[0], ent[1][0]):   # one nested entity per item; sub-entities in name order
                _leaf_values(name, [leaf], out)
    return out


def detect_bbox_order(records: list, max_tokens: int = 4000) -> str:
    """VRDU boxes are [page, a, b, c, d] (normalised). Decide between (x0,y0,x1,y1) and (x0,x1,y0,y1):
    the right order gives non-negative extents and near-constant token heights."""
    import statistics
    ha, hb, ok_a, ok_b, n = [], [], True, True, 0
    for rec in records:
        for page in rec["ocr"]["pages"]:
            H = page["dimension"]["height"]
            for t in page["tokens"]:
                _, a, b, c, d = t["bbox"][:5]
                ok_a &= c >= a - 1e-6 and d >= b - 1e-6
                ok_b &= b >= a - 1e-6 and d >= c - 1e-6
                ha.append((d - b) * H)
                hb.append((d - c) * H)
                n += 1
                if n >= max_tokens:
                    break
    if ok_a and not ok_b:
        return "xyxy"
    if ok_b and not ok_a:
        return "xxyy"

    def cv(v):
        v = [x for x in v if x > 0]
        return statistics.pstdev(v) / statistics.mean(v) if len(v) > 1 else 9e9
    return "xyxy" if cv(ha) <= cv(hb) else "xxyy"


def _to_box(bb, order, W, H):
    _, a, b, c, d = bb[:5]
    x0, y0, x1, y1 = (a, b, c, d) if order == "xyxy" else (a, c, b, d)
    return clip_box([x0 * W, y0 * H, x1 * W, y1 * H], W, H)


def _vrdu_template(corpus_short, rec, values):
    stem = Path(rec["filename"]).stem
    if corpus_short == "fara":
        parts = stem.split("_")
        return f"vrdu/fara/{slug('_'.join(parts[-2:]) if len(parts) >= 3 else stem)}"
    props = [v[1] for v in values if v[0] == "property" and v[1].strip()]
    return f"vrdu/deepform/{slug(props[0])}" if props else f"vrdu/deepform/doc_{slug(stem, 40)}"


def iter_vrdu(root, stats: Counter, corpora=("registration-form", "ad-buy-form"), ocr: str = "provided",
              max_docs: int | None = None, all_pages: bool = False, ocr_cache=None, bbox_order: str = "auto"):
    """Yield one unified page document at a time (VRDU is large - never hold it all in memory)."""
    root = Path(root)
    for corpus in corpora:
        main = root / corpus / "main"
        if not main.is_dir():
            stats[f"vrdu_missing_{corpus}"] += 1
            continue
        short = VRDU_CORPORA.get(corpus, slug(corpus))
        f, path = _open_jsonl(main)
        order = bbox_order
        buffered = []
        if order == "auto":
            for _ in range(5):
                line = f.readline()
                if not line:
                    break
                buffered.append(json.loads(line))
            order = detect_bbox_order(buffered)
            stats[f"vrdu_{short}_bbox_order_{order}"] += 1

        def records():
            yield from buffered
            for line in f:
                if line.strip():
                    yield json.loads(line)
        n_docs = 0
        for rec in records():
            if max_docs is not None and n_docs >= max_docs:
                break
            n_docs += 1
            values = parse_vrdu_annotations(rec.get("annotations"))
            tmpl = _vrdu_template(short, rec, values)
            pdf = main / rec.get("file_path", f"pdfs/{rec['filename']}")
            by_page = defaultdict(list)
            for name, text, bb, segs in values:
                if text.strip():
                    by_page[int(bb[0])].append((name, text, segs))
            pages = rec["ocr"]["pages"]
            for page in pages:
                k = int(page.get("page_id", pages.index(page)))
                if not all_pages and not by_page.get(k):
                    stats["vrdu_pages_without_gold_skipped"] += 1
                    continue
                W, H = int(page["dimension"]["width"]), int(page["dimension"]["height"])
                if ocr == "provided":
                    words, starts = [], []
                    for t in page["tokens"]:
                        txt = (t.get("text") or "").strip()
                        if not txt:
                            continue
                        words.append({"text": txt, "box": _to_box(t["bbox"], order, W, H)})
                        starts.append(t["segments"][0][0] if t.get("segments") else -1)
                    lines = []
                    for ln in page.get("lines", []):
                        s0, s1 = ln["segments"][0] if ln.get("segments") else (-1, -1)
                        ids = [i for i, st in enumerate(starts) if s0 <= st < s1]
                        if ids:
                            lines.append(ids)
                    covered = {i for l in lines for i in l}
                    lines += [[i] for i in range(len(words)) if i not in covered]
                    gold = []
                    for name, text, segs in by_page.get(k, []):
                        wid = [i for i, st in enumerate(starts) for s0, s1 in segs if s0 <= st < s1]
                        gold.append({"field": name, "text": text.strip(), "word_ids": sorted(set(wid))})
                else:  # docTR on the rendered page
                    from .ocr import ocr_images, render_pdf
                    cache = Path(ocr_cache or (root / "_doctr_cache")) / short
                    cache.mkdir(parents=True, exist_ok=True)
                    cj = cache / f"{slug(Path(rec['filename']).stem, 100)}_p{k:03d}.json"
                    if cj.exists():
                        res = json.loads(cj.read_text(encoding="utf-8"))
                    else:
                        imgs = render_pdf(pdf, cache / "pages", max_pages=k + 1)
                        res = ocr_images([imgs[k]])[0]
                        cj.write_text(json.dumps(res), encoding="utf-8")
                    words, lines, W, H = res["words"], res["lines"], res["width"], res["height"]
                    gold = [{"field": name, "text": text.strip()} for name, text, _ in by_page.get(k, [])]
                seen, dedup = set(), []
                for g in gold:
                    key = (g["field"], alnum(g["text"]))
                    if key not in seen:
                        seen.add(key)
                        dedup.append(g)
                if not words:
                    stats["vrdu_page_without_words"] += 1
                    continue
                stats[f"vrdu_{short}_pages"] += 1
                stats[f"vrdu_{short}_gold_values"] += len(dedup)
                yield _finish({
                    "doc_id": f"vrdu/{short}/{slug(Path(rec['filename']).stem, 100)}#p{k}", "source": f"vrdu-{short}",
                    "split_hint": "test", "lang": "en", "template_id": tmpl, "image_path": None,
                    "pdf_path": str(pdf.resolve()) if pdf.exists() else None, "page_index": k,
                    "width": W, "height": H, "words": words, "lines": lines, "entities": [], "links": [],
                    "header_links": [], "annotation": "values", "link_supervision": False,
                    "gold_values": dedup, "ocr": ocr, "split": "unseen_test"}, stats)
            stats[f"vrdu_{short}_docs"] += 1
        f.close()
