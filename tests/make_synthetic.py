"""Generate small synthetic multi-template datasets in the REAL on-disk formats
(FUNSD folders, XFUND json, custom template folders) to test the pipeline end to end.

Each template is a different layout: different field names, positions, and
key->value directions (right / below / left), plus edge cases the real data has:
multi-line values, a key with two values, a value with no key, blank answers,
duplicated and reversed links, header->key links.
"""
import argparse
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw

FIELDS = ["Name", "Date", "Address", "Phone", "Account No", "Amount", "Signature", "City", "PIN Code",
          "Email", "Occupation", "Branch", "IFSC", "Nominee", "Age", "Gender", "Policy", "Remarks"]
WORDS = ["alpha", "bravo", "delta", "kumar", "sharma", "pune", "mumbai", "road", "lane", "12", "345",
         "2026", "nagar", "colony", "bank", "india", "rao", "devi", "singh", "plot"]


def make_template(rng, tid, W, H):
    n = rng.randint(5, 8)
    names = rng.sample(FIELDS, n)
    direction = rng.choice(["right", "below", "mixed"])
    col_x = rng.choice([60, 90, 140])
    y = rng.choice([120, 160, 200])
    fields = []
    for i, nm in enumerate(names):
        d = direction if direction != "mixed" else rng.choice(["right", "below", "left"])
        x = col_x if d != "left" else col_x + 380
        fields.append({"key": nm + ":", "x": x + rng.randint(-10, 10), "y": y, "dir": d,
                       "lines": 2 if (i == 2 and rng.random() < 0.7) else 1})
        y += 110 if d == "below" else 70
        if y > H - 150:
            break
    return {"id": tid, "header": f"FORM {tid.upper()}", "fields": fields, "two_value_field": rng.random() < 0.5}


def word_box(x, y, text, h=22):
    return [x, y, x + 12 * len(text), y + h]


def render_doc(rng, tpl, W, H, img_path):
    ents, eid = [], 0
    im = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(im)

    def add(label, words, linking=None):
        nonlocal eid
        e = {"id": eid, "label": label, "text": " ".join(w["text"] for w in words),
             "box": [min(w["box"][0] for w in words) if words else 0, 0, 0, 0],
             "words": words, "linking": linking or []}
        ents.append(e)
        eid += 1
        return e

    jit = lambda: rng.randint(-4, 4)  # noqa: E731
    hx, hy = 220 + jit(), 50 + jit()
    header = add("header", [{"text": t, "box": word_box(hx + 70 * i, hy, t)} for i, t in enumerate(tpl["header"].split())])
    for f in tpl["fields"]:
        kx, ky = f["x"] + jit(), f["y"] + jit()
        kwords, cx = [], kx
        for t in f["key"].split():
            kwords.append({"text": t, "box": word_box(cx, ky, t)})
            cx += 12 * len(t) + 10
        key = add("question", kwords)
        key["linking"].append([header["id"], key["id"]])
        header["linking"].append([header["id"], key["id"]])
        n_vals = 2 if (tpl["two_value_field"] and f is tpl["fields"][0]) else 1
        for vi in range(n_vals):
            if f["dir"] == "right":
                vx, vy = cx + 25 + vi * 170, ky
            elif f["dir"] == "below":
                vx, vy = kx + vi * 170, ky + 34
            else:  # value to the LEFT of its key
                vx, vy = kx - 330 + vi * 0, ky + vi * 30
            vwords = []
            for line in range(f["lines"]):
                xx = vx
                for _ in range(rng.randint(1, 3)):
                    t = rng.choice(WORDS)
                    vwords.append({"text": t, "box": word_box(xx, vy + 28 * line, t)})
                    xx += 12 * len(t) + 10
            if rng.random() < 0.05:  # blank answer: words with empty text (real FUNSD has these)
                vwords = [{"text": " ", "box": word_box(vx, vy, "   ")}]
            val = add("answer", vwords)
            pair = [key["id"], val["id"]] if rng.random() < 0.5 else [val["id"], key["id"]]  # either order
            key["linking"].append(pair)
            val["linking"].append(pair)  # duplicated, as in FUNSD
    add("answer", [{"text": "orphan", "box": word_box(W - 200, H - 80, "orphan")}])  # value with no key
    add("other", [{"text": "page", "box": word_box(40, H - 40, "page")}, {"text": "1", "box": word_box(100, H - 40, "1")}])
    for e in ents:
        for w in e["words"]:
            col = {"question": (0, 0, 0), "answer": (40, 40, 160), "header": (0, 0, 0), "other": (90, 90, 90)}[e["label"]]
            d.text((w["box"][0], w["box"][1] + 4), w["text"], fill=col)
    for e in ents:  # FUNSD-style entity boxes
        ws = e["words"]
        e["box"] = [min(w["box"][0] for w in ws), min(w["box"][1] for w in ws),
                    max(w["box"][2] for w in ws), max(w["box"][3] for w in ws)]
    img_path.parent.mkdir(parents=True, exist_ok=True)
    im.save(img_path)
    return ents


# ---------------------------------------------------------------- receipts + VRDU stand-ins
MERCHANTS = ["TAN WOON YANN BOOK TA K SDN BHD", "OJC MARKETING SDN BHD", "GARDENIA BAKERIES SDN BHD",
             "PASARAYA BORONG PINTAR", "SANYU STATIONERY SHOP"]


def make_sroie(rng, root):
    for split, n in (("train", 18), ("test", 6)):
        for d in ("box", "entities", "img"):
            (root / split / d).mkdir(parents=True, exist_ok=True)
        for k in range(n):
            stem = f"X{split[:2].upper()}{k:05d}"
            m = rng.choice(MERCHANTS)
            addr = f"NO {rng.randint(1, 99)}, JALAN {rng.choice(['SAGU', 'BAYU', 'PUTRA'])} {rng.randint(1, 20)}, JOHOR"
            date = f"{rng.randint(1, 28):02d}/{rng.randint(1, 12):02d}/2018"
            total = f"{rng.randint(1, 300)}.{rng.randint(0, 99):02d}"
            rows = [m, addr, f"DATE: {date}", f"ITEM {rng.choice(WORDS).upper()} 1 X {total}", f"TOTAL: RM{total}",
                    f"CASH {total}", "THANK YOU"]
            W, H, y = 420, 40 + 30 * len(rows), 20
            im = Image.new("RGB", (W, H), "white")
            dr = ImageDraw.Draw(im)
            lines = []
            for r in rows:
                x0, x1 = 20, min(W - 10, 20 + 9 * len(r))
                lines.append(f"{x0},{y},{x1},{y},{x1},{y + 20},{x0},{y + 20},{r}")
                dr.text((x0, y + 4), r, fill="black")
                y += 30
            (root / split / "box" / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
            (root / split / "entities" / f"{stem}.txt").write_text(json.dumps(
                {"company": m, "date": date, "address": addr, "total": total}, indent=4), encoding="utf-8")
            im.save(root / split / "img" / f"{stem}.jpg")


WR_CLASSES = ["Ignore", "Store_name_value", "Store_name_key", "Store_addr_value", "Store_addr_key", "Tel_value",
              "Tel_key", "Date_value", "Date_key", "Time_value", "Time_key", "Prod_item_value", "Prod_item_key",
              "Prod_quantity_value", "Prod_quantity_key", "Prod_price_value", "Prod_price_key", "Subtotal_value",
              "Subtotal_key", "Tax_value", "Tax_key", "Tips_value", "Tips_key", "Total_value", "Total_key", "Others"]


def make_wildreceipt(rng, root):
    root.mkdir(parents=True, exist_ok=True)
    (root / "class_list.txt").write_text("\n".join(f"{i} {c}\t" if i % 5 == 0 else f"{i} {c}"
                                                  for i, c in enumerate(WR_CLASSES)) + "\n", encoding="utf-8")
    cid = {c: i for i, c in enumerate(WR_CLASSES)}
    stores = ["SAFEWAY", "CHOEUN KOREAN", "WALMART", "TARGET"]
    for split, n in (("train", 16), ("test", 6)):
        out = []
        for k in range(n):
            store = rng.choice(stores)
            fn = f"image_files/Image_{k % 3}/{k}/{split}_{k}.jpeg"
            W, H = 400, 520
            segs = [(store, "Store_name_value"), ("Tel:", "Tel_key"), (f"555-{rng.randint(1000, 9999)}", "Tel_value"),
                    ("Date", "Date_key"), (f"0{rng.randint(1, 9)}/1{rng.randint(0, 9)}/2019", "Date_value"),
                    ("ITEM", "Prod_item_key"), ("PRICE", "Prod_price_key")]
            for _ in range(rng.randint(2, 4)):
                segs += [(rng.choice(WORDS).upper(), "Prod_item_value"), (f"{rng.randint(1, 30)}.99", "Prod_price_value")]
            segs += [("TOTAL", "Total_key"), (f"{rng.randint(10, 99)}.00", "Total_value"), ("THANK YOU", "Others"),
                     ("@#", "Ignore")]
            anns, y, x = [], 20, 30
            for text, lab in segs:
                w = 10 * len(text)
                if lab.endswith("_value") and lab not in ("Store_name_value",):
                    bx = 230
                else:
                    bx = x
                anns.append({"box": [bx, y, bx + w, y, bx + w, y + 18, bx, y + 18], "text": text, "label": cid[lab]})
                if not lab.endswith("_key") or lab in ("Prod_item_key",):
                    y += 28
            im = Image.new("RGB", (W, H), "white")
            dr = ImageDraw.Draw(im)
            for a in anns:
                dr.text((a["box"][0], a["box"][1] + 3), a["text"], fill="black")
            (root / fn).parent.mkdir(parents=True, exist_ok=True)
            im.save(root / fn)
            out.append(json.dumps({"file_name": fn, "height": H, "width": W, "annotations": anns}))
        (root / f"{split}.txt").write_text("\n".join(out) + "\n", encoding="utf-8")


def make_vrdu(rng, root, order="xyxy"):
    """dataset.jsonl in VRDU's layout: ocr.text, ocr.pages[{page_id, dimension, tokens, lines}], annotations."""
    specs = {"registration-form": ("FARA", {"registration_num": ("Registration No.", lambda: str(rng.randint(1000, 9999))),
                                             "registrant_name": ("Name of Registrant", lambda: f"{rng.choice(MERCHANTS).title()}"),
                                             "file_date": ("Date", lambda: f"{rng.randint(1, 12)}/{rng.randint(1, 28)}/2019")}),
             "ad-buy-form": ("DeepForm", {"advertiser": ("Advertiser:", lambda: rng.choice(["Biden For President", "NRCC"])),
                                          "contract_num": ("Contract #", lambda: str(rng.randint(100000, 999999))),
                                          "property": ("Station", lambda: rng.choice(["KMSP", "WTVF"]))})}
    for corpus, (dname, fields) in specs.items():
        main = root / corpus / "main"
        (main / "pdfs").mkdir(parents=True, exist_ok=True)
        meta = {"dataset_name": dname, "entity_name_to_match_func": {f: "GeneralStringMatch" for f in fields}}
        if corpus == "ad-buy-form":   # table line items, annotated in VRDU's nested format
            meta["entity_name_to_match_func"].update({"channel": "GeneralStringMatch", "sub_amount": "PriceMatch"})
            meta["entity_appearance_pattern"] = {**{f: "unrepeated" for f in fields},
                                                 "channel": "line_item", "sub_amount": "line_item"}
        (main / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
        recs = []
        for k in range(6):
            stem = f"2019010{k}_Some Registrant_Amendment_Amendment" if corpus == "registration-form" else f"doc-{k:04d}"
            W, H = 1700, 2200
            text, pages, annotations = "", [], []
            imgs = []
            for pg in range(2):
                tokens, lines, page_start = [], [], len(text)
                im = Image.new("RGB", (W // 2, H // 2), "white")
                dr = ImageDraw.Draw(im)
                y = 0.08 + 0.02 * k
                header = ["U.S. Department of Justice", f"Page {pg + 1}"]
                rows = [(h, None) for h in header]
                if pg == 0:
                    rows += [(f"{lab} {gen()}", f) for f, (lab, gen) in fields.items()]
                    if corpus == "ad-buy-form":
                        rows += [("Channel Amount", None)]
                        rows += [(f"{rng.choice(['KMSP', 'WTVF', 'KSTP'])} ${rng.randint(1, 9)}00.00", "__li__")
                                 for _ in range(2)]
                else:
                    rows += [("Signature of Registrant", None)]
                for row, field in rows:
                    line_start = len(text)
                    x = 0.1
                    spans = []
                    for wi, wd in enumerate(row.split()):
                        s0 = len(text)
                        spans.append((wd, s0, x))
                        text += wd + " "
                        w = 0.012 * len(wd)
                        a, b, c, d = (x, y, x + w, y + 0.012) if order == "xyxy" else (x, x + w, y, y + 0.012)
                        tokens.append({"bbox": [pg, a, b, c, d], "segments": [[s0, s0 + len(wd) + 1]],
                                       "text": wd + " ", "orientation": "PAGE_UP"})
                        dr.text((x * W / 2, y * H / 2), wd, fill="black")
                        x += w + 0.01
                    text = text[:-1] + "\n"
                    lines.append({"bbox": [pg, 0.1, y, x, y + 0.012], "segments": [[line_start, len(text)]],
                                  "text": text[line_start:], "orientation": "PAGE_UP"})
                    if field == "__li__":
                        subs = [[wd, [pg, sx, y, sx + 0.012 * len(wd), y + 0.012], [[s0, s0 + len(wd)]]]
                                for wd, s0, sx in spans]
                        annotations.append([["channel", "sub_amount"], [subs]])
                    elif field:
                        lab = fields[field][0]
                        vstart = line_start + len(lab) + 1
                        vtext = text[vstart:len(text)]
                        annotations.append([field, [[vtext, [pg, 0.3, y, 0.6, y + 0.012], [[vstart, len(text)]]]]])
                    y += 0.05
                pages.append({"page_id": pg, "dimension": {"height": H, "width": W}, "segment": [page_start, len(text)],
                              "tokens": tokens, "lines": lines, "paragraphs": lines, "blocks": lines})
                imgs.append(im)
            imgs[0].save(main / "pdfs" / f"{stem}.pdf", save_all=True, append_images=imgs[1:])
            recs.append(json.dumps({"filename": f"{stem}.pdf", "file_path": f"./pdfs/{stem}.pdf",
                                    "ocr": {"text": text, "pages": pages}, "annotations": annotations}))
        (main / "dataset.jsonl").write_text("\n".join(recs) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--funsd-templates", type=int, default=14)
    ap.add_argument("--docs-per-template", type=int, default=4)
    ap.add_argument("--vrdu-order", default="xyxy", choices=["xyxy", "xxyy"])
    args = ap.parse_args()
    rng = random.Random(args.seed)
    root = Path(args.out)
    W, H = 760, 1000

    # FUNSD format: train/test folders; templates spread across both (like the real thing)
    k = 0
    for t in range(args.funsd_templates):
        tpl = make_template(rng, f"f{t}", W, H)
        for j in range(args.docs_per_template if t % 3 else 1):  # some singleton templates
            split = "testing_data" if rng.random() < 0.25 else "training_data"
            stem = f"funsd{k:04d}"
            ents = render_doc(rng, tpl, W, H, root / "FUNSD/dataset" / split / "images" / f"{stem}.png")
            ann = root / "FUNSD/dataset" / split / "annotations" / f"{stem}.json"
            ann.parent.mkdir(parents=True, exist_ok=True)
            ann.write_text(json.dumps({"form": ents}), encoding="utf-8")
            k += 1
    for split in ("training_data", "testing_data"):
        (root / "FUNSD/dataset" / split / "annotations").mkdir(parents=True, exist_ok=True)
        (root / "FUNSD/dataset" / split / "images").mkdir(parents=True, exist_ok=True)

    # XFUND format: same template shared by two languages (must be clustered together)
    for lang in ("de", "fr"):
        for split in ("train", "val"):
            docs = []
            for t in range(4):
                tpl = make_template(random.Random(1000 + t), f"x{t}", W, H)
                for j in range(2):
                    did = f"{lang}_{split}_{t}_{j}"
                    fname = did + ".jpg"
                    ents = render_doc(rng, tpl, W, H, root / "XFUND" / f"{lang}.{split}" / fname)
                    docs.append({"id": did, "img": {"fname": fname, "width": W, "height": H}, "document": ents})
            (root / "XFUND" / f"{lang}.{split}.json").write_text(
                json.dumps({"lang": lang, "split": split, "documents": docs}), encoding="utf-8")

    # custom: one folder per template
    for t in range(5):
        tpl = make_template(rng, f"c{t}", W, H)
        for j in range(4):
            stem = f"doc{j}"
            tdir = root / "custom" / f"form_{t}"
            ents = render_doc(rng, tpl, W, H, tdir / "images" / f"{stem}.png")
            (tdir / "annotations").mkdir(parents=True, exist_ok=True)
            (tdir / "annotations" / f"{stem}.json").write_text(json.dumps({"form": ents}), encoding="utf-8")
    make_sroie(rng, root / "SROIE2019")
    make_wildreceipt(rng, root / "wildreceipt")
    make_vrdu(rng, root / "vrdu", order=args.vrdu_order)
    print("synthetic data written to", root)


if __name__ == "__main__":
    main()
