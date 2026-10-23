"""Convert FUNSD, XFUND and the self-curated set into the unified schema (kvx.schema).

All three use the same entity format ("FUNSD-style"):
    {"id", "label": question|answer|header|other, "words": [{"text", "box"}], "linking": [[a, b], ...]}
so they share one converter. Every dropped item is counted in `stats` - nothing
disappears silently.
"""
from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path
from urllib.parse import unquote

from PIL import Image

from .schema import clip_box, validate_doc

LABEL2ROLE = {"question": "key", "answer": "value", "header": "header", "other": "other",
              "key": "key", "value": "value"}  # Label Studio exports may use Key / Value
IMG_EXT = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def _image_size(path: Path | None):
    if path is not None and path.exists():
        with Image.open(path) as im:
            return im.size
    return None


def _size_from_words(form):
    xs = [wd["box"][2] for e in form for wd in e.get("words", [])] or [1000]
    ys = [wd["box"][3] for e in form for wd in e.get("words", [])] or [1000]
    return int(max(xs)) + 1, int(max(ys)) + 1


def convert_form(form: list, w: float, h: float, stats: Counter):
    """FUNSD-style entity list -> (words, entities, links, header_links)."""
    words, entities, old2new = [], [], {}
    for ent in form:
        role = LABEL2ROLE.get(str(ent.get("label", "")).lower())
        if role is None:
            stats["drop_unknown_label"] += 1
            continue
        if ent["id"] in old2new:
            stats["drop_duplicate_entity_id"] += 1
            continue
        wids = []
        for wd in ent.get("words", []):
            text = (wd.get("text") or "").strip()
            if not text:
                stats["drop_empty_word"] += 1
                continue
            wids.append(len(words))
            words.append({"text": text, "box": clip_box(wd["box"], w, h)})
        if not wids:
            stats["drop_empty_entity"] += 1
            continue
        new_id = len(entities)
        old2new[ent["id"]] = new_id
        entities.append({"id": new_id, "role": role, "word_ids": wids})

    raw_pairs = set()
    for ent in form:
        for pair in ent.get("linking", []) or []:
            if len(pair) != 2 or pair[0] == pair[1]:
                continue
            raw_pairs.add(frozenset(pair))

    links, header_links = set(), set()
    for pair in raw_pairs:
        a, b = tuple(pair)
        if a not in old2new or b not in old2new:
            stats["drop_link_to_dropped_entity"] += 1
            continue
        na, nb = old2new[a], old2new[b]
        ra, rb = entities[na]["role"], entities[nb]["role"]
        if (ra, rb) == ("key", "value"):
            links.add((na, nb))
        elif (ra, rb) == ("value", "key"):
            links.add((nb, na))
        elif (ra, rb) == ("header", "key"):
            header_links.add((na, nb))
        elif (ra, rb) == ("key", "header"):
            header_links.add((nb, na))
        else:
            stats["drop_link_" + "-".join(sorted((ra, rb)))] += 1
    return words, entities, sorted(links), sorted(header_links)


def _make_doc(doc_id, source, split_hint, lang, template_id, img_path, w, h, form, stats):
    words, entities, links, header_links = convert_form(form, w, h, stats)
    doc = {
        "doc_id": doc_id,
        "source": source,
        "split_hint": split_hint,
        "lang": lang,
        "template_id": template_id,
        "image_path": str(img_path.resolve()) if img_path is not None and img_path.exists() else None,
        "width": int(w),
        "height": int(h),
        "words": words,
        "entities": entities,
        "links": [list(p) for p in links],
        "header_links": [list(p) for p in header_links],
    }
    if not words:
        stats["drop_doc_without_words"] += 1
        return None
    errs = validate_doc(doc)
    if errs:
        raise ValueError(f"{doc_id}: invalid after conversion: {errs[:3]}")
    return doc


def _find_image(folder: Path, stem: str):
    for ext in IMG_EXT:
        p = folder / (stem + ext)
        if p.exists():
            return p
    return None


# --------------------------------------------------------------------------- FUNSD
def load_funsd(root: str | Path, stats: Counter) -> list[dict]:
    """root = the FUNSD 'dataset' folder holding training_data/ and testing_data/."""
    root = Path(root)
    splits = {"train": root / "training_data", "test": root / "testing_data"}
    docs = []
    for split, d in splits.items():
        ann_dir = d / "annotations"
        if not ann_dir.is_dir():
            raise FileNotFoundError(f"FUNSD: missing {ann_dir} (point --funsd at the 'dataset' folder)")
        for ann in sorted(ann_dir.glob("*.json")):
            form = json.loads(ann.read_text(encoding="utf-8"))["form"]
            img = _find_image(d / "images", ann.stem)
            size = _image_size(img)
            if size is None:
                stats["funsd_image_missing"] += 1
                size = _size_from_words(form)
            doc = _make_doc(f"funsd/{ann.stem}", "funsd", split, "en", None, img, size[0], size[1], form, stats)
            if doc:
                docs.append(doc)
    return docs


# --------------------------------------------------------------------------- XFUND
def load_xfund(root: str | Path, langs: list[str], stats: Counter) -> list[dict]:
    """Finds XFUND files named {lang}.{split}.json anywhere under root (e.g. de.train.json, fr.val.json).

    Official XFUND JSON: {"documents": [{"id", "img": {"fname", "width", "height"},
                                         "document": [FUNSD-style entities]}]}
    Images are looked for in {lang}.{split}/ next to the json, next to the json itself, or in images/.
    If nothing loads, the error lists every json file seen and why it was skipped.
    """
    root = Path(root)
    files = sorted(root.rglob("*.json"))
    if not files:
        raise FileNotFoundError(f"XFUND: no .json files anywhere under {root}")
    docs, skipped = [], []
    name_re = re.compile(r"^([a-z]{2})\.(train|val|test)$", re.IGNORECASE)
    for jf in files:
        m = name_re.match(jf.stem)
        if not m:
            skipped.append(f"{jf.relative_to(root)}  (name is not <lang>.<split>.json)")
            continue
        lang, split = m.group(1).lower(), m.group(2).lower()
        if langs and lang not in langs:
            skipped.append(f"{jf.relative_to(root)}  (language '{lang}' not in --xfund-langs {','.join(langs)})")
            continue
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            skipped.append(f"{jf.relative_to(root)}  (unreadable json: {e})")
            continue
        items = data.get("documents") if isinstance(data, dict) else None
        if not items:
            keys = list(data)[:6] if isinstance(data, dict) else type(data).__name__
            skipped.append(f"{jf.relative_to(root)}  (no 'documents' list; top-level: {keys})")
            continue
        base = jf.parent
        n_before = len(docs)
        for item in items:
            img_meta = item.get("img", {})
            fname = img_meta.get("fname", "")
            img = None
            for cand in (base / f"{lang}.{split}" / fname, base / fname, base / "images" / fname):
                if fname and cand.exists():
                    img = cand
                    break
            if img is None:
                stats["xfund_image_missing"] += 1
            size = None
            if not (img_meta.get("width") and img_meta.get("height")):
                size = _image_size(img) or _size_from_words(item["document"])
            w = img_meta.get("width") or size[0]
            h = img_meta.get("height") or size[1]
            doc = _make_doc(f"xfund/{lang}/{item['id']}", "xfund", split, lang, None, img, w, h,
                            item["document"], stats)
            if doc:
                docs.append(doc)
        stats[f"xfund_loaded_{lang}.{split}"] += len(docs) - n_before
    if not docs:
        listing = "\n  ".join(skipped[:30]) or "(none)"
        raise FileNotFoundError(f"XFUND: 0 documents loaded from {root}. Files seen:\n  {listing}")
    for msg in skipped:
        stats["xfund_file_skipped"] += 1
    return docs


# -------------------------------------------------------------------------- custom
def load_custom(root: str | Path, stats: Counter, lang: str = "en") -> list[dict]:
    """Self-curated forms, one folder per template, FUNSD-format annotations:

        root/<template_name>/annotations/<doc>.json   {"form": [...]}
        root/<template_name>/images/<doc>.png|jpg

    The folder name IS the template id, so these templates never need clustering.
    """
    root = Path(root)
    docs = []
    tdirs = sorted(p for p in root.iterdir() if p.is_dir())
    if not tdirs:
        raise FileNotFoundError(f"custom: no template folders in {root}")
    for tdir in tdirs:
        ann_dir = tdir / "annotations"
        if not ann_dir.is_dir():
            stats["custom_folder_without_annotations"] += 1
            continue
        for ann in sorted(ann_dir.glob("*.json")):
            form = json.loads(ann.read_text(encoding="utf-8"))["form"]
            img = _find_image(tdir / "images", ann.stem)
            size = _image_size(img)
            if size is None:
                stats["custom_image_missing"] += 1
                size = _size_from_words(form)
            doc = _make_doc(f"custom/{tdir.name}/{ann.stem}", "custom", "all", lang,
                            f"custom/{tdir.name}", img, size[0], size[1], form, stats)
            if doc:
                docs.append(doc)
    return docs


def dataset_stats(docs: list[dict]) -> Counter:
    """Structural statistics - the numbers to report on the dataset slide."""
    s = Counter()
    for d in docs:
        src = d["source"]
        s[f"{src}:docs"] += 1
        s[f"{src}:annotation_{d.get('annotation', 'full')}"] += 1
        s[f"{src}:unsupervised_words"] += len(d.get("ignore_words", []) or [])
        s[f"{src}:words"] += len(d["words"])
        s[f"{src}:links"] += len(d["links"])
        roles = [e["role"] for e in d["entities"]]
        for r in roles:
            s[f"{src}:entities_{r}"] += 1
        vk = Counter(k for k, _ in d["links"])
        kv = Counter(v for _, v in d["links"])
        for i, r in enumerate(roles):
            if r == "key" and i not in vk:
                s[f"{src}:key_without_value"] += 1
            if r == "value" and i not in kv:
                s[f"{src}:value_without_key"] += 1
        s[f"{src}:key_with_multiple_values"] += sum(1 for c in vk.values() if c > 1)
        s[f"{src}:value_with_multiple_keys"] += sum(1 for c in kv.values() if c > 1)
    return s


# -------------------------------------------------------------------- Label Studio
_LS_HASH = re.compile(r"^[0-9a-f]{8}-", re.IGNORECASE)  # Label Studio prefixes uploads with a hash


def _ls_image(ref, images_dir):
    if not ref:
        return None, None
    name = Path(unquote(str(ref)).split("?")[0]).name
    clean = _LS_HASH.sub("", name)
    if images_dir:
        for cand in (name, clean):
            if (Path(images_dir) / cand).exists():
                return Path(images_dir) / cand, Path(clean).stem
        found = _find_image(Path(images_dir), Path(clean).stem)
        if found:
            return found, Path(clean).stem
    return None, Path(clean).stem


def _ls_split_words(text, box):
    """Label Studio stores one transcription per region; split the region box over its lines and
    words in proportion to character counts (an approximation, counted in the stats)."""
    lines = [l for l in text.replace("\r", "").split("\n") if l.strip()]
    if not lines:
        return []
    x0, y0, x1, y1 = box
    lh = (y1 - y0) / len(lines)
    words = []
    for li, line in enumerate(lines):
        toks = line.split()
        total = sum(len(t) for t in toks) + max(0, len(toks) - 1)
        unit = (x1 - x0) / total if total else 0.0
        cx = x0
        for t in toks:
            words.append({"text": t, "box": [cx, y0 + li * lh, cx + unit * len(t), y0 + (li + 1) * lh]})
            cx += unit * (len(t) + 1)
    return words


def read_template_csv(path) -> dict:
    """CSV with columns image,template (image = file name or stem)."""
    out = {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            out[Path(row["image"].strip()).stem] = row["template"].strip()
    return out


def load_labelstudio(export_path, images_dir, stats: Counter, template_csv=None, lang: str = "en") -> list[dict]:
    """Label Studio JSON export -> unified docs (source "custom").

    Labelling setup: RectangleLabels with Key/Value/Header/Other (or Question/Answer), a per-region
    TextArea with the transcription, and relations drawn between key and value regions.
    The template of each image comes from task data field "template" or from --labelstudio-templates.
    """
    tasks = json.loads(Path(export_path).read_text(encoding="utf-8"))
    template_of = read_template_csv(template_csv) if template_csv else {}
    docs = []
    for task in tasks:
        anns = [a for a in (task.get("annotations") or task.get("completions") or []) if not a.get("was_cancelled")]
        if not anns:
            stats["ls_task_without_annotation"] += 1
            continue
        data = task.get("data", {})
        img, stem = _ls_image(data.get("image") or data.get("ocr"), images_dir)
        stem = stem or f"task{task.get('id')}"
        regions, texts, rels = {}, {}, []
        w = h = None
        for r in anns[-1].get("result", []):
            typ, val = r.get("type"), r.get("value", {})
            if typ in ("rectanglelabels", "labels"):
                labels = val.get("rectanglelabels") or val.get("labels") or []
                if labels:
                    if abs(float(val.get("rotation", 0) or 0)) > 1e-6:
                        stats["ls_rotated_region_box_used_unrotated"] += 1
                    w, h = r.get("original_width", w), r.get("original_height", h)
                    regions[r["id"]] = (labels[0], val)
            elif typ == "textarea":
                t = val.get("text") or []
                texts[r["id"]] = "\n".join(t) if isinstance(t, list) else str(t)
            elif typ == "relation":
                rels.append((r.get("from_id"), r.get("to_id")))
        size = _image_size(img)
        if size:
            w, h = size
        else:
            stats["ls_image_missing"] += 1
        if not w or not h:
            stats["ls_task_without_size"] += 1
            continue
        form, id_of = [], {}
        for rid, (label, val) in regions.items():
            text = texts.get(rid, "").strip()
            if not text:
                stats["ls_region_without_text"] += 1
                continue
            box = [val["x"] / 100 * w, val["y"] / 100 * h,
                   (val["x"] + val["width"]) / 100 * w, (val["y"] + val["height"]) / 100 * h]
            words = _ls_split_words(text, box)
            if len(words) > 1:
                stats["ls_region_word_boxes_approximated"] += 1
            id_of[rid] = len(form)
            form.append({"id": len(form), "label": label.lower(), "words": words, "linking": []})
        for a, b in rels:
            if a in id_of and b in id_of:
                form[id_of[a]]["linking"].append([id_of[a], id_of[b]])
            else:
                stats["ls_relation_to_missing_region"] += 1
        tid = data.get("template") or template_of.get(stem)
        if not tid:
            raise ValueError(f"Label Studio image {stem}: no template id (add data.template or a row in the CSV)")
        doc = _make_doc(f"custom/{tid}/{stem}", "custom", "all", lang, f"custom/{tid}", img, w, h, form, stats)
        if doc:
            docs.append(doc)
    return docs
