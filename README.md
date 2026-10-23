# Cross-template key-value extraction from scanned forms
IE 643 course project: generalising to form templates with zero same-template training data.

## The idea
Every form prints its own keys. So instead of asking "where is field X?" (which needs a schema that unseen
templates don't share), the model answers two template-agnostic questions:

1. **Role tagging**: is each word part of a KEY, a VALUE, a HEADER, or nothing? (BIO tags on LayoutLMv3)
2. **Linking**: which key does each value answer? Every value takes a softmax over the keys in its chunk
   plus a "no key" option, scored by `s(k, v) = MLP([e_k ; e_v ; e_k ⊙ e_v ; φ(g(k, v))])`, where `g(k, v)` is
   **relative** geometry in units of word height (no absolute page position, which is how a model memorises a template).

Loss: `L = L_tag + λ · L_link`. A template never seen in training goes through exactly the same code path.

## Data: what trains, what tests
| Dataset | Role | What it supervises |
|---|---|---|
| FUNSD (199 forms) | train / val / seen-test | keys, values, headers, key→value links |
| WildReceipt (1,739 receipts) | train / val / seen-test | `X_key`/`X_value` classes → keys, values; links derived (each value → the key of its category) |
| SROIE (973 receipts) | train / val / seen-test | **values only** (company, date, address, total, found by string matching). Every other word is left *unsupervised*, because SROIE marks no keys. No link supervision. |
| VRDU (registration + ad-buy forms) | **held-out test only** | gold field values per page; VRDU marks no printed keys |

* Templates: FUNSD templates are induced by layout + key-text clustering; SROIE's template is the merchant; WildReceipt's
  is the store name; VRDU's is the FARA form type or the ad-buy TV station.
* Training sources are split by template into `train` / `val` (model selection) / `seen_test` (unseen documents of
  seen templates). **All of VRDU is `unseen_test`**; it is never read during training.
* Sampling is source-balanced, so the ~2,700 receipts don't drown FUNSD.

## OCR
Documents without annotations (uploads in the interface, and optionally VRDU) are read by **docTR** on the GPU,
which returns word-level boxes grouped in lines; PDFs are rendered page by page. The preferred models are
`fast_base` (detector) + `parseq` (recogniser). If the `parseq` weights are not on disk and cannot be downloaded,
the best model already in the docTR cache is used instead (e.g. `crnn_vgg16_bn`), and the interface shows which pair is
running. `scripts/fetch_ocr_weights.py` downloads and verifies the preferred weights, or prints the URL and the exact
folder for a manual browser download. VRDU ships its own OCR, which is the default for the benchmark
(`--ocr provided`); `--ocr doctr` runs the whole pipeline on our OCR instead.

## Metrics
| Metric | Where | Meaning |
|---|---|---|
| **value_recall** | all datasets | gold values found as predicted VALUE spans (text match; numbers must match exactly) **← cross-dataset headline** |
| **kv_field_recall** | SROIE, VRDU | value found *and* linked to a printed key naming that field (`kvx/field_aliases.yaml`) |
| entity_f1 / link_f1 / e2e_pair_f1 | FUNSD, WildReceipt | spans; linking on gold spans; exact end-to-end pairs |
| value_acc, value_precision | FUNSD, WildReceipt | chosen key correct; predicted values that are real values |
| VRDU field P / R / F1 (step 7) | VRDU | the benchmark's own scoring: each predicted pair is named as a field via its printed key (aliases), values compared with VRDU's type-specific match functions (`meta.json`); micro, macro, per field, line items |

`-` in a report means "not measurable on these documents", never zero. The seen − unseen gap comes with a 95% CI
from resampling templates.

```
kvx/      schema · loaders (FUNSD, XFUND, custom, Label Studio) · loaders_extra (SROIE, WildReceipt, VRDU) · ocr (docTR)
          templates · encoding · data · model · train · predict · metrics · evaluate · analysis · viz · runtime · config
scripts/  00_check_env · 01_build_data · 01b_build_vrdu · check_alignment · 02_template_split · 03_viz_gold · 04_train
          05_evaluate · 06_failure_analysis · fetch_ocr_weights · inspect_datasets
app/      Flask interface: upload image/PDF → OCR → key-value pairs drawn on the page; held-out examples; results
tests/    test_core.py (unit) · smoke_test.py (whole pipeline offline) · synthetic data + tiny backbone builders
```

---

## What is not in this repository

`git` carries the code, the config, the report and nothing else. Four things are deliberately absent
because they are large, reproducible, or not ours to redistribute:

| Missing | Size | How to get it |
| --- | --- | --- |
| `data/` | several GB | public corpora: FUNSD, SROIE (ICDAR 2019 Task 3), WildReceipt, VRDU. Sources are cited in the report. |
| `cache/` | ~300 MB | rebuilt by steps 1 and 1b below, from `data/` |
| `runs/` | ~2 GB | four trained checkpoints, published as [GitHub release assets](https://github.com/MudiamSukesh/cross-template-kv-extraction/releases/tag/v1.0); or retrain with step 4 |
| `.venv/` | ~5 GB | `python -m venv .venv` then the install lines below |

A fresh clone therefore runs the test suite immediately, and reproduces every reported number once
`data/` is in place.

## Run order (Windows PowerShell, from the project folder)
```powershell
cd C:\Users\sukes\Documents\finalproject
.\.venv\Scripts\Activate.ps1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
$D = "C:\Users\sukes\Documents\finalproject\data"

python tests\smoke_test.py ; Remove-Item -Recurse -Force smoke_work   # must say SMOKE TEST PASSED
python scripts\fetch_ocr_weights.py                                    # optional: get parseq (else cached crnn is used)
python scripts\00_check_env.py                                         # GPU, LayoutLMv3, and a real OCR test

python scripts\01_build_data.py --funsd $D\FUNSD --sroie $D\SROIE2019 --wildreceipt $D\wildreceipt --out cache\docs.jsonl
python scripts\01b_build_vrdu.py --vrdu $D\vrdu --out cache\vrdu.jsonl
python scripts\check_alignment.py --docs cache\vrdu.jsonl                # must say ALIGNED
python scripts\check_alignment.py --docs cache\docs.jsonl                # must say ALIGNED
python scripts\02_template_split.py --docs cache\docs.jsonl --out cache\split
python scripts\04_train.py --config configs\base.yaml
python scripts\05_evaluate.py --run runs\base
python scripts\06_failure_analysis.py --run runs\base
python scripts\07_vrdu_benchmark.py --run runs\base --vrdu $D\vrdu    # VRDU-style field P / R / F1 (no model run)
python app\app.py --run runs\base                                      # http://127.0.0.1:5000
```
The in-domain unseen-template run (this is what makes entity / link / end-to-end pair F1 measurable on
templates the model has never seen - the base split holds out only VRDU, which annotates values only):
```powershell
python scripts\02_template_split.py --docs cache\docs.jsonl --out cache\split_u --unseen-frac 0.15
python scripts\04_train.py --set data.docs=cache/split_u/docs_split.jsonl train.out_dir=runs\unseen15
python scripts\05_evaluate.py      --run runs\unseen15
python scripts\06_failure_analysis.py --run runs\unseen15
```
Optional: VRDU through our own OCR (slow the first time; results are cached):
```powershell
python scripts\01b_build_vrdu.py --vrdu $D\vrdu --ocr doctr --out cache\vrdu_doctr.jsonl
python scripts\05_evaluate.py --run runs\base --splits unseen_test --test-docs cache\vrdu_doctr.jsonl
```

**What to check**
* Step 1: SROIE `*_matched_exact / _fuzzy / _unmatched` counters; WildReceipt `value_without_key`.
* Step 1b: pages / gold values per corpus and the detected `bbox_order`; then `check_alignment.py` measures
  automatically that the boxes sit on the printed text (share of boxes containing ink vs a shifted control).
* Step 2: `Leakage assertions: PASSED`; FUNSD montages show one layout each.
* Step 4: per-epoch `val value-rec / entity / link / e2e`; `best.pt` is chosen on validation templates.
* Step 5: `eval\report.txt`, the per-field tables, and `qualitative\`.
* Step 6: `analysis\failure_report.md` - sections E-G are the held-out (VRDU) analysis.
* Step 7: `eval\vrdu_benchmark.txt` - VRDU precision / recall / F1 per corpus and field, scored from the saved predictions.
  Passing `--split-files` restricts scoring to VRDU's own split lists; check the `documents N` line per corpus
  before reading the numbers, because a glob that matches no file for a corpus yields a section of zeros from
  zero documents, which is not a result.

## Interface
`python app\app.py --run runs\base`, then http://127.0.0.1:5000
* **Extract** - drop an image or PDF; docTR reads it; the page shows the extracted key-value pairs as boxes.
  Hover a row to light up its key, its value and the link; toggle the word-label layer (key / value / header /
  other with counts); per-page OCR and model timings; page buttons for multi-page PDFs.
* **Held-out examples** - VRDU pages (unseen templates) and seen-template test documents, with every gold value
  marked found / missed and the page's scores.
* **Results** - the measured table from step 5: seen vs unseen with 95% CIs, the gap, per source, per field.
The header shows which checkpoint, backbone, device and OCR models are running. Models load once; requests are
served one at a time (one GPU).

## Ablations (one key each)
```powershell
python scripts\04_train.py --set train.out_dir=runs\visual    model.visual=true
python scripts\04_train.py --set train.out_dir=runs\absgeom   model.link_geometry=absolute
python scripts\04_train.py --set train.out_dir=runs\noaug     augment.enabled=false
python scripts\04_train.py --set train.out_dir=runs\nobalance data.balance_sources=false
python scripts\04_train.py --set train.out_dir=runs\nosroie   "data.train_sources=[funsd,wildreceipt]"
```

## Limitations to state in the report
* **VRDU differs from the training data in domain as well as template** (government and TV-ad forms vs.
  receipts and FUNSD), so the seen → VRDU gap mixes both effects.
* SROIE supervises values only; its unlabelled words carry no signal (they are neither keys nor "other").
* kv_field_recall depends on the alias list in `kvx/field_aliases.yaml`; publish it with the numbers.
* Training pages use the datasets' own text segments as units; docTR lines at inference group text a little differently.
* SROIE and WildReceipt templates are defined by the printed merchant / store name; spelling variants of one shop
  become separate templates, which can make the validation split slightly optimistic (VRDU is unaffected).
* Values whose key is not printed (a lone date, a station address) can be found but not keyed.
* LayoutLMv3-base is English-only.
