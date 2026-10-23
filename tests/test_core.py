"""Unit tests with hand-computed answers. Run:  python tests/test_core.py   (pytest also works)"""
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from kvx.loaders import convert_form  # noqa: E402
from kvx.metrics import doc_counts, f1, similarity, summarize  # noqa: E402
from kvx.model import geometry_features, link_loss  # noqa: E402
from kvx.predict import bio_decode  # noqa: E402
from kvx.schema import TAG2ID, relation_direction, validate_doc  # noqa: E402
from kvx.templates import check_split, split_templates  # noqa: E402


def W(t, x, y):
    return {"text": t, "box": [x, y, x + 40, y + 20]}


def form_fixture():
    return [
        {"id": 10, "label": "header", "words": [W("FORM", 100, 10)], "linking": [[10, 11]]},
        {"id": 11, "label": "question", "words": [W("Name:", 10, 50)], "linking": [[11, 12], [10, 11]]},
        {"id": 12, "label": "answer", "words": [W("Ravi", 60, 50), W("Kumar", 105, 50)], "linking": [[11, 12]]},
        {"id": 13, "label": "question", "words": [W("Date:", 10, 90)], "linking": [[14, 13], [15, 13]]},
        {"id": 14, "label": "answer", "words": [W(" ", 60, 90)], "linking": [[14, 13]]},       # blank -> dropped
        {"id": 15, "label": "answer", "words": [W("12/3", 110, 90)], "linking": [[15, 13], [13, 15]]},
        {"id": 16, "label": "other", "words": [W("p1", 10, 300)], "linking": []},
        {"id": 17, "label": "answer", "words": [W("orphan", 200, 300)], "linking": []},
    ]


def make_doc():
    st = Counter()
    words, ents, links, hl = convert_form(form_fixture(), 400, 400, st)
    return {"doc_id": "t/1", "source": "t", "template_id": "t/a", "width": 400, "height": 400, "words": words,
            "entities": ents, "links": [list(x) for x in links], "header_links": [list(x) for x in hl]}, st


def test_convert_form():
    d, st = make_doc()
    assert validate_doc(d) == [], validate_doc(d)
    roles = [e["role"] for e in d["entities"]]
    assert roles == ["header", "key", "value", "key", "value", "other", "value"], roles
    assert d["links"] == [[1, 2], [3, 4]], d["links"]           # dedup + reversed link oriented key->value
    assert d["header_links"] == [[0, 1]]
    assert st["drop_empty_word"] == 1 and st["drop_empty_entity"] == 1 and st["drop_link_to_dropped_entity"] == 1


def test_duplicate_entity_id_leaves_no_orphan_words():
    form = form_fixture() + [{"id": 11, "label": "question", "words": [W("Dup:", 300, 50)], "linking": []}]
    st = Counter()
    words, ents, _, _ = convert_form(form, 400, 400, st)
    assert st["drop_duplicate_entity_id"] == 1
    assert sorted(w for e in ents for w in e["word_ids"]) == list(range(len(words))), "orphan words"


def test_labelstudio_loader(tmp=None):
    import json
    import tempfile
    from PIL import Image
    from kvx.loaders import load_labelstudio
    tmp = Path(tempfile.mkdtemp())
    (tmp / "img").mkdir()
    Image.new("RGB", (1000, 500), "white").save(tmp / "img" / "pan_form_01.png")
    def region(rid, label, x, y, w, h, text):
        geo = {"x": x, "y": y, "width": w, "height": h, "rotation": 0}
        return [{"id": rid, "type": "rectanglelabels", "original_width": 1000, "original_height": 500,
                 "value": dict(geo, rectanglelabels=[label])},
                {"id": rid, "type": "textarea", "value": dict(geo, text=[text])}]
    result = region("a", "Key", 5, 10, 10, 4, "Full Name") + region("b", "Value", 20, 10, 30, 8, "Ravi\nKumar")
    result.append({"type": "relation", "from_id": "a", "to_id": "b", "direction": "right"})
    export = [{"id": 1, "data": {"image": "/data/upload/1/3f9a2b1c-pan_form_01.png", "template": "pan"},
               "annotations": [{"result": result}]}]
    (tmp / "export.json").write_text(json.dumps(export), encoding="utf-8")
    st = Counter()
    docs = load_labelstudio(tmp / "export.json", tmp / "img", st)
    d = docs[0]
    assert d["doc_id"] == "custom/pan/pan_form_01" and d["template_id"] == "custom/pan"
    assert [w["text"] for w in d["words"]] == ["Full", "Name", "Ravi", "Kumar"]
    assert d["links"] == [[0, 1]] and validate_doc(d) == []
    kb = d["words"][0]["box"]
    assert abs(kb[0] - 50) < 1e-6 and abs(kb[1] - 50) < 1e-6, kb       # 5% of 1000, 10% of 500
    assert d["words"][3]["box"][1] > d["words"][2]["box"][1], "second line placed below the first"


def perfect_pred(d):
    return {"pred_entities": [{"role": e["role"], "word_ids": e["word_ids"]} for e in d["entities"]
                              if e["role"] in ("key", "value", "header")],
            "pred_links": [], "gold_mode_links": [list(x) for x in d["links"]],
            "gold_mode": {str(v): {"key": k} for k, v in d["links"]}}


def test_metrics_perfect_and_partial():
    d, _ = make_doc()
    p = perfect_pred(d)
    idx = {tuple(e["word_ids"]): i for i, e in enumerate(p["pred_entities"])}
    ents = d["entities"]
    p["pred_links"] = [[idx[tuple(ents[k]["word_ids"])], idx[tuple(ents[v]["word_ids"])]] for k, v in d["links"]]
    s = summarize([doc_counts(d, p)])
    for m in ("value_recall", "entity_f1", "link_f1", "e2e_pair_f1", "e2e_fuzzy_f1", "value_acc", "value_precision"):
        assert abs(s[m] - 1.0) < 1e-9, (m, s[m])
    # drop one of two links and split one value: hand-computed results
    p2 = perfect_pred(d)
    p2["gold_mode_links"] = [[1, 2]]
    p2["gold_mode"] = {"2": {"key": 1}, "4": {"key": None}}
    p2["pred_entities"] = [e for e in p2["pred_entities"] if e["word_ids"] != ents[2]["word_ids"]]
    p2["pred_entities"] += [{"role": "value", "word_ids": [ents[2]["word_ids"][0]]},
                            {"role": "value", "word_ids": [ents[2]["word_ids"][1]]}]
    c = doc_counts(d, p2)
    assert c["link"] == (1, 1, 2)                       # P = 1, R = 0.5
    assert c["entity_value"] == (2, 4, 3)               # 2 of 4 predicted values exact, 3 gold values
    assert c["value_acc"] == (1, 0, 2)
    assert abs(f1(*c["link"])[0] - 2 / 3) < 1e-9


def test_fuzzy_similarity():
    assert similarity("ravi kumar", "ravi kumar") == 1.0
    assert abs(similarity("kumar", "kumaar") - (1 - 1 / 6)) < 1e-9
    assert similarity("", "x") == 0.0


def test_bio_decode():
    t = {i: TAG2ID[x] for i, x in enumerate(["B-KEY", "I-KEY", "B-VALUE", "I-VALUE", "I-VALUE", "O", "I-VALUE", "B-VALUE", "I-KEY"])}
    ents = bio_decode(list(range(9)), t)
    assert [(e["role"], e["word_ids"]) for e in ents] == [
        ("key", [0, 1]), ("value", [2, 3, 4]), ("value", [6]), ("value", [7]), ("key", [8])], ents


def test_geometry_invariance():
    kb = torch.tensor([[100., 100., 180., 120.], [300., 400., 380., 420.]])
    vb = torch.tensor([[200., 100., 300., 120.], [100., 140., 220., 160.], [500., 600., 540., 640.]])
    g = geometry_features(kb, vb, 20.0, "relative")
    assert g.shape == (3, 2, 12)
    assert torch.allclose(g, geometry_features(kb + 37, vb + 37, 20.0, "relative"), atol=1e-5), "translation"
    assert torch.allclose(g, geometry_features(kb * 1.5, vb * 1.5, 30.0, "relative"), atol=1e-5), "scale"
    ga = geometry_features(kb, vb, 20.0, "absolute")
    assert ga.shape == (3, 2, 16) and not torch.allclose(ga, geometry_features(kb + 37, vb + 37, 20.0, "absolute"))
    assert geometry_features(kb, vb, 20.0, "none").abs().sum() == 0


def test_geometry_same_on_any_page_format():
    from kvx.schema import iso_box
    k = [100, 100, 200, 120]
    v_right, v_below = [260, 100, 360, 120], [100, 180, 200, 200]   # 60 px gap right, 60 px gap below
    for W, H in ((800, 1000), (1000, 707), (612, 792)):
        kb = torch.tensor([iso_box(k, W, H)])
        vb = torch.tensor([iso_box(v_right, W, H), iso_box(v_below, W, H)])
        g = geometry_features(kb, vb, iso_box([0, 0, 0, 20], W, H)[3], "relative")
        gap_x_right, gap_y_below = g[0, 0, 2], g[1, 0, 3]
        assert torch.allclose(gap_x_right, gap_y_below, atol=1e-5), (W, H, gap_x_right, gap_y_below)


def test_direction():
    assert relation_direction([0, 0, 50, 20], [60, 0, 120, 20]) == "right"
    assert relation_direction([0, 0, 50, 20], [0, 30, 60, 50]) == "below"
    assert relation_direction([60, 0, 120, 20], [0, 0, 50, 20]) == "left"


def test_link_loss():
    logits = torch.tensor([[5.0, 0.0, 0.0], [0.0, 0.0, 5.0], [0.0, 0.0, 0.0]])  # 2 keys + null
    loss, n = link_loss(logits, [[0], [], None])
    assert n == 2
    expect = -(torch.log_softmax(logits[0], 0)[0] + torch.log_softmax(logits[1], 0)[2])
    assert torch.allclose(loss, expect)
    multi, _ = link_loss(torch.zeros(1, 3), [[0, 1]])          # two gold keys: p = 2/3
    assert torch.allclose(multi, -torch.log(torch.tensor(2 / 3)))


def test_split_no_leak_and_assertion():
    docs = []
    for t in range(10):
        for j in range(3 if t % 2 else 1):
            docs.append({"doc_id": f"s/{t}_{j}", "template_id": f"s/t{t}"})
    a = split_templates(docs, 0.2, 0.1, 0.3, seed=1)
    check_split(docs, a)
    tr = {d["template_id"] for d in docs if a[d["doc_id"]] == "train"}
    un = {d["template_id"] for d in docs if a[d["doc_id"]] == "unseen_test"}
    assert tr and un and not (tr & un) and len(un) >= 2
    forced = split_templates(docs, 0.2, 0.1, 0.3, seed=1, forced_unseen=("s/t1", "s/t3"))
    check_split(docs, forced)
    bad = dict(forced)
    bad["s/1_0"] = "train"                       # one document of unseen template s/t1 moved into train
    try:
        check_split(docs, bad)
    except AssertionError as e:
        assert "LEAK" in str(e), e
    else:
        raise AssertionError("leak not detected")


def test_chunking_keeps_units_whole():
    from kvx.encoding import Encoder, doc_units

    class FakeTok:  # 2 tokens per word
        pad_token_id = 1

        def __call__(self, texts, boxes=None, **kw):
            class E(dict):
                def word_ids(s):
                    return [i for i in range(len(texts)) for _ in range(2)]
            return E()
    d, _ = make_doc()
    enc = Encoder(FakeTok(), max_tokens=8)   # budget 6 tokens = 3 words
    units = doc_units(d)
    chunks = enc.chunk_units(d, units)
    flat = [w for ch in chunks for u in ch for w in u]
    assert sorted(flat) == list(range(len(d["words"]))), "every word exactly once"
    for ch in chunks:
        assert sum(2 * len(u) for u in ch) <= 6
        for u in ch:
            assert u in units, "units are never split when they fit"


def test_sroie_value_matching():
    from kvx.loaders_extra import find_value_spans
    words = ["TOTAL:", "RM9.00", "CASH", "9.00", "BOOK", "TA", ".K", "(TAMAN", "DAYA)", "SDN", "BHD", "19.00"]
    st = Counter()
    spans = find_value_spans(words, "9.00", set(), all_occurrences=True, stats=st, tag="t")
    assert spans == [[1], [3]], spans                       # RM prefix allowed, 19.00 is NOT a match
    sp = find_value_spans(words, "BOOK TA .K (TAMAN DAYA) SDN BHD", set(), False, st, "c")
    assert sp == [[4, 5, 6, 7, 8, 9, 10]], sp
    fz = find_value_spans(words, "BOOK TA K TAMAN DAYA SDN BND", set(), False, st, "f")  # OCR typo -> fuzzy
    assert fz == [[4, 5, 6, 7, 8, 9, 10]] and st["f_matched_fuzzy"] == 1, (fz, st)
    assert find_value_spans(words, "77.00", set(), False, st, "u") == [] and st["u_unmatched"] == 1


def test_vrdu_annotation_parsing():
    from kvx.loaders_extra import parse_vrdu_annotations
    anns = [["registration_num", [["3712\n", [0, .4, .3, .5, .34], [[10, 15]]]]],
            ["line_item", [[["channel", [["KMSP", [1, .1, .2, .3, .4], [[5, 9]]]]],
                            ["sub_amount", [["$500", [1, .5, .2, .6, .4], [[20, 24]]]]]]]]]
    out = parse_vrdu_annotations(anns)
    assert [(n, t, b[0]) for n, t, b, _ in out] == [("registration_num", "3712\n", 0), ("channel", "KMSP", 1),
                                                    ("sub_amount", "$500", 1)], out


def test_vrdu_official_nested_format():
    """VRDU's own format for line items: the entity name is a LIST of sub-field names (benchmark_utils.py)."""
    from kvx.loaders_extra import parse_vrdu_annotations
    anns = [["advertiser", [["NRCC", [0, .1, .1, .2, .12], [[0, 4]]]]],
            [["channel", "sub_amount"], [[["KMSP", [1, .1, .5, .2, .52], [[10, 14]]],
                                          ["$500", [1, .3, .5, .4, .52], [[15, 19]]]]]]]
    assert [(n, t, b[0]) for n, t, b, _ in parse_vrdu_annotations(anns)] == [
        ("advertiser", "NRCC", 0), ("channel", "KMSP", 1), ("sub_amount", "$500", 1)]


def test_vrdu_benchmark_scoring():
    from kvx.vrdu_eval import doc_extractions, fields_for_key, match_doc, prf
    funcs = {"advertiser": "GeneralStringMatch", "gross_amount": "PriceMatch", "flight_from": "DateMatch",
             "flight_to": "DateMatch", "channel": "GeneralStringMatch", "sub_amount": "PriceMatch"}
    pattern = {f: "unrepeated" for f in funcs}
    pattern.update(channel="line_item", sub_amount="line_item")
    meta = {"funcs": funcs, "pattern": pattern,
            "groups": {"unrepeated": ["advertiser", "gross_amount", "flight_from", "flight_to"], "repeated": [],
                       "line_item": ["channel", "sub_amount"]}}
    aliases = {"advertiser": ["advertiser"], "gross_amount": ["gross amount", "total"], "flight_from": ["flight", "from"],
               "flight_to": ["flight", "to"], "channel": ["channel", "station"], "sub_amount": ["amount"]}
    assert fields_for_key("Advertiser:", "NRCC", 1, meta, aliases) == ["advertiser"]
    assert fields_for_key("Gross Amount", "$1,500", 1, meta, aliases) == ["gross_amount"]      # longest alias wins
    assert fields_for_key("Flight dates", "10/01/20 - 10/07/20", 1, meta, aliases) == ["flight_from", "flight_to"]
    assert fields_for_key("Remarks", "x", 1, meta, aliases) == []
    fara = {"funcs": {"file_date": "DateMatch", "signer_name": "NameMatch"}, "pattern": {}, "groups": {}}
    fal = {"file_date": ["date"], "signer_name": ["name", "signature"]}
    assert fields_for_key("Date of signature", "06/05/2019", 1, fara, fal) == ["file_date"]     # type beats length
    assert fields_for_key("Signature", "John Smith", 1, fara, fal) == ["signer_name"]
    P = (0, 0, 0)
    exts = [{"field": "advertiser", "text": "Wrong Co", "p": 0.4, "pos": P},
            {"field": "advertiser", "text": "NRCC.", "p": 0.9, "pos": P},               # most probable is kept
            {"field": "gross_amount", "text": "$1,500", "p": 0.8, "pos": P},
            {"field": "flight_from", "text": "10/01/20", "p": 0.7, "pos": P},
            {"field": "flight_to", "text": "10/7/2020", "p": 0.7, "pos": P},
            {"field": "channel", "text": "KMSP", "p": 0.6, "pos": (0, 10, 1)},
            {"field": "sub_amount", "text": "500.00", "p": 0.6, "pos": (0, 10, 5)},
            {"field": "channel", "text": "WTVF", "p": 0.6, "pos": (0, 11, 1)},
            {"field": "sub_amount", "text": "$800", "p": 0.6, "pos": (0, 11, 5)}]
    ex = doc_extractions(exts, meta)
    assert ("advertiser", "NRCC.") in ex and ("advertiser", "Wrong Co") not in ex
    assert (("channel", "sub_amount"), ("KMSP", "500.00")) in ex
    gold = [("advertiser", [("NRCC", 0)]), ("gross_amount", [("1500.00", 0)]), ("flight_from", [("10/1/2020", 0)]),
            ("flight_to", [("10/07/20", 0)]), (("sub_amount", "channel"), [(("$500", 0), ("KMSP", 0))]),
            (("channel", "sub_amount"), [(("WTVF", 0), ("$700", 0))])]
    c, g, x = match_doc(gold, ex, meta)
    assert (c, g, x) == (5, 6, 6), (c, g, x)            # second row's amount is wrong ($800 vs $700)
    assert abs(prf(c, g, x)["f1"] - 5 / 6) < 1e-9
    assert match_doc(gold, ex, meta, {"line_item"}) == (1, 2, 2)


def test_value_only_document_metrics_and_tags():
    from kvx.encoding import doc_units, gold_word_tags
    words = [W("Total", 0, 0), W("RM9.00", 60, 0), W("Thanks", 0, 40)]
    doc = {"doc_id": "s/1", "source": "sroie", "template_id": "s", "width": 200, "height": 100, "words": words,
           "lines": [[0, 1], [2]], "entities": [{"id": 0, "role": "value", "word_ids": [1]}], "links": [],
           "annotation": "values", "ignore_words": [0, 2], "gold_values": [{"field": "total", "text": "9.00"}]}
    assert validate_doc(doc) == []
    assert gold_word_tags(doc) == [-100, TAG2ID["B-VALUE"], -100]
    assert doc_units(doc) == [[0, 1], [2]]
    pred = {"pred_entities": [{"role": "key", "word_ids": [0]}, {"role": "value", "word_ids": [1]}],
            "pred_links": [[0, 1]], "gold_mode_links": [], "gold_mode": {}}
    c = doc_counts(doc, pred, 0.8, aliases={"total": ["total"]})
    assert "link" not in c and "entity_key" not in c         # not measurable on value-only documents
    s = summarize([c])
    assert s["link_f1"] is None and s["entity_f1"] is None
    assert s["value_recall"] == 1.0 and s["kv_field_recall"] == 1.0   # "RM9.00" counts as the value "9.00"
    from kvx.metrics import value_match
    assert value_match("1 000 00", "1000 00", 0.8) and not value_match("19 00", "9 00", 0.8)
    assert not value_match("rm19 00", "9 00", 0.8)
    assert value_match("6/5/2019", "06/05/2019", 0.8) and value_match("06-05-19", "6/5/2019", 0.8)
    assert not value_match("6/6/2019", "06/05/2019", 0.8)


def test_multiline_value_concatenation():
    words = [W("Address:", 0, 0), W("4614", 60, 0), W("Center", 110, 0), W("Chicago", 60, 30), W("IL", 110, 30)]
    doc = {"doc_id": "v/1", "source": "vrdu-deepform", "template_id": "v", "width": 300, "height": 100, "words": words,
           "entities": [], "links": [], "annotation": "values",
           "gold_values": [{"field": "tv_address", "text": "4614 Center\nChicago IL"}]}
    pred = {"pred_entities": [{"role": "key", "word_ids": [0]}, {"role": "value", "word_ids": [1, 2]},
                              {"role": "value", "word_ids": [3, 4]}],
            "pred_links": [[0, 1], [0, 2]], "gold_mode_links": [], "gold_mode": {}}
    assert summarize([doc_counts(doc, pred, 0.8, aliases={})])["value_recall"] == 1.0


def test_tag_loss_without_supervised_words_is_zero():
    from kvx.model import tag_loss
    logits = torch.randn(2, 5, 7, requires_grad=True)
    assert tag_loss(logits, torch.full((2, 5), -100)).item() == 0.0
    y = torch.full((2, 5), -100)
    y[0, 1] = 3
    expect = torch.nn.functional.cross_entropy(logits[0, 1:2], torch.tensor([3]))
    assert torch.allclose(tag_loss(logits, y), expect)


def test_bootstrap_handles_missing_metrics():
    from kvx.metrics import bootstrap_ci
    counts = {f"d{i}": {"value_rec": (i % 2, 0, 1), "kv_field": (0, 0, 0), "pred_values": (1, 0, 1)} for i in range(10)}
    ci = bootstrap_ci(counts, {f"d{i}": f"t{i % 5}" for i in range(10)}, ["value_recall", "link_f1"], B=200)
    assert ci["link_f1"] is None and ci["value_recall"] is not None


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok ", t.__name__)
    print(f"\nall {len(tests)} tests passed")
