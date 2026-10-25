"""One-off: make the counterfactual probes edit the page image alongside the boxes."""
from pathlib import Path

ANALYSIS = Path("kvx/analysis.py")
PREDICT = Path("kvx/predict.py")

NEW_MAKE_PROBE = '''def make_probe(doc: dict, probe: str, with_image: bool = False):
    """Edited deep copy of doc, plus a page image edited to match (None where pixels cannot follow)."""
    from PIL import Image
    d = copy.deepcopy(doc)
    W, H = d["width"], d["height"]
    h = median_word_height(d)
    im = page_image(d) if with_image else None

    if probe in ("identity", "reading_order"):
        pass
    elif probe == "shift_scale":
        s, tx, ty = 0.8, 0.1 * W, 0.1 * H
        for w in d["words"]:
            b = w["box"]
            w["box"] = [b[0] * s + tx, b[1] * s + ty, b[2] * s + tx, b[3] * s + ty]
        if im is not None:
            im = im.transform((int(W), int(H)), Image.AFFINE,
                              (1 / s, 0, -tx / s, 0, 1 / s, -ty / s),
                              resample=Image.BILINEAR, fillcolor="white")
    elif probe == "mirror_x":
        for w in d["words"]:
            b = w["box"]
            w["box"] = [W - b[2], b[1], W - b[0], b[3]]
        if im is not None:
            im = im.transpose(Image.FLIP_LEFT_RIGHT)
    elif probe in ("value_below", "value_right"):
        moved = set()
        for k, v in d["links"]:
            if v in moved:
                continue
            kb, vb = entity_box(d, d["entities"][k]), entity_box(d, d["entities"][v])
            if probe == "value_below":
                dx, dy = kb[0] - vb[0], kb[3] + 0.4 * h - vb[1]
            else:
                dx, dy = kb[2] + 1.0 * h - vb[0], kb[1] - vb[1]
            _move_entity(d, d["entities"][v], dx, dy)
            moved.add(v)
        im = None
    elif probe == "key_text_masked":
        for e in d["entities"]:
            if e["role"] == "key":
                for w in e["word_ids"]:
                    d["words"][w]["text"] = "xxxx"
        im = None
    else:
        raise ValueError(probe)

    if im is None:
        d["image_path"] = None
    return d, im


'''

NEW_RUN_PROBES = '''def run_probes(model, encoder, docs, device, adt, fuzzy=0.8, probes=PROBES) -> dict:
    with_image = encoder.image_processor is not None
    out = {}
    for probe in probes:
        counts = []
        for d in docs:
            pd, im = make_probe(d, probe, with_image)
            pred = predict_doc(model, encoder, pd, device, adt,
                               use_gold_units=(probe != "reading_order"), image=im)
            counts.append(doc_counts(pd, pred, fuzzy))
        s = summarize(counts)
        out[probe] = {k: s[k] for k in ("value_recall", "value_acc", "link_f1", "entity_f1", "e2e_pair_f1")}
    return out


'''

OLD_FOOTER = '"Probes run without the page image (edited geometry would contradict the pixels), so always compare",'
NEW_FOOTER = ('"Probes that translate, scale or mirror the whole page carry the page image edited to match; "\n'
              '          "value_below, value_right and key_text_masked drop it, because pixels cannot follow a "\n'
              '          "per-entity move. A text-and-layout run has no image either way. Always compare",')


def span(text, start, end, new):
    i = text.index(start)
    j = text.index(end, i)
    return text[:i] + new + text[j:]


def once(text, old, new):
    if text.count(old) != 1:
        raise SystemExit("anchor not found exactly once: " + old[:70])
    return text.replace(old, new)


def backup(p, text):
    Path(str(p) + ".bak").write_text(text, encoding="utf-8")


a = ANALYSIS.read_text(encoding="utf-8")
backup(ANALYSIS, a)
a = span(a, "def make_probe(", "PROBES = [", NEW_MAKE_PROBE)
a = span(a, "def run_probes(", "def sample_test_docs(", NEW_RUN_PROBES)
if a.count(OLD_FOOTER) == 1:
    a = a.replace(OLD_FOOTER, NEW_FOOTER)
    print("footer wording updated")
else:
    print("WARNING: footer line not matched, edit it by hand")
ANALYSIS.write_text(a, encoding="utf-8")

p = PREDICT.read_text(encoding="utf-8")
backup(PREDICT, p)
p = once(p,
         "def predict_doc(model, encoder, doc: dict, device, amp_dtype=None, use_gold_units: bool = True) -> dict:",
         "def predict_doc(model, encoder, doc: dict, device, amp_dtype=None,\n"
         "                use_gold_units: bool = True, image=None) -> dict:")
p = once(p,
         "    image = page_image(doc) if encoder.image_processor is not None else None",
         "    if image is None and encoder.image_processor is not None:\n        image = page_image(doc)")
PREDICT.write_text(p, encoding="utf-8")

print("patched kvx/analysis.py and kvx/predict.py (.bak written beside each)")
