"""OCR: text + word bounding boxes for documents that come without annotations
(uploads in the interface, and the VRDU test PDFs).

Engine: docTR (Mindee), PyTorch, runs on the GPU.
  detector   fast_base  - docTR's default text detector
  recognizer parseq     - docTR's most accurate recognizer
docTR returns WORD-level boxes grouped into lines and blocks, which is what LayoutLMv3 needs.
Weights are downloaded once and cached. If the preferred weights are not on disk and cannot be
downloaded, the best model already in the cache is used instead (resolve_archs), and the interface
shows which pair is running. scripts/fetch_ocr_weights.py downloads/verifies the preferred weights.

Each docTR line becomes one "unit" (kvx.encoding.doc_units), in docTR's block -> line order.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

DEFAULT_DET = os.environ.get("KVX_OCR_DET", "fast_base")
DEFAULT_RECO = os.environ.get("KVX_OCR_RECO", "parseq")
DET_FALLBACKS = ["fast_base", "db_resnet50", "linknet_resnet50", "fast_small", "db_resnet34"]
RECO_FALLBACKS = ["parseq", "crnn_vgg16_bn", "vitstr_base", "master", "sar_resnet31", "crnn_mobilenet_v3_large"]
PDF_SCALE = 2.0          # 144 dpi; docTR is trained on ~150 dpi pages
IMG_EXT = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp")


def use_system_certificates():
    """Let Python's downloads trust the Windows/macOS certificate store (needed when antivirus or a proxy
    inspects HTTPS - the usual cause of SSL errors while docTR downloads its weights)."""
    try:
        import truststore
        truststore.inject_into_ssl()
        return True
    except Exception:  # noqa: BLE001 - optional
        return False


def _registry() -> dict:
    import importlib
    import pkgutil
    import doctr.models as dm
    cfgs = {}
    for info in pkgutil.walk_packages(dm.__path__, dm.__name__ + "."):
        if info.name.endswith(".pytorch") and (".detection." in info.name or ".recognition." in info.name):
            try:
                mod = importlib.import_module(info.name)
            except Exception:  # noqa: BLE001
                continue
            cfgs.update(getattr(mod, "default_cfgs", {}) or {})
    return cfgs


def _cache_dir() -> Path:
    return Path(os.environ.get("DOCTR_CACHE_DIR", Path.home() / ".cache" / "doctr")) / "models"


def weight_urls(det_arch: str = DEFAULT_DET, reco_arch: str = DEFAULT_RECO) -> dict:
    """{arch: (download url, cache file path)} for the two docTR models (read from docTR's own registry)."""
    cfgs = _registry()
    out = {}
    for arch in (det_arch, reco_arch):
        url = (cfgs.get(arch) or {}).get("url")
        if not url:
            raise KeyError(f"unknown docTR architecture {arch}")
        out[arch] = (url, _cache_dir() / url.rpartition("/")[-1].split("&")[0])
    return out


@lru_cache(maxsize=64)
def weights_ok(arch: str) -> bool:
    """Weights for this architecture are on disk and pass docTR's checksum."""
    import hashlib
    url = (_registry().get(arch) or {}).get("url")
    if not url:
        return False
    p = _cache_dir() / url.rpartition("/")[-1].split("&")[0]
    if not p.is_file():
        return False
    prefix = p.stem.rsplit("-", 1)[-1]
    return hashlib.sha256(p.read_bytes()).hexdigest().startswith(prefix)


@lru_cache(maxsize=1)
def resolve_archs(det_arch: str = DEFAULT_DET, reco_arch: str = DEFAULT_RECO) -> tuple[str, str, str]:
    """-> (det, reco, note). Preferred models if their weights are cached; otherwise the first cached
    fallback; otherwise the preferred ones (docTR will try to download them)."""
    if os.environ.get("KVX_OCR_UNTRAINED") == "1":
        return det_arch, reco_arch, "untrained weights (offline test mode)"
    picked, notes = [], []
    for pref, fallbacks in ((det_arch, DET_FALLBACKS), (reco_arch, RECO_FALLBACKS)):
        if weights_ok(pref):
            picked.append(pref)
            continue
        alt = next((a for a in fallbacks if a != pref and weights_ok(a)), None)
        if alt:
            picked.append(alt)
            notes.append(f"{pref} weights not on disk - using cached {alt}")
        else:
            picked.append(pref)
    return picked[0], picked[1], "; ".join(notes)


@lru_cache(maxsize=2)
def get_predictor(det_arch: str | None = None, reco_arch: str | None = None):
    """Cached docTR predictor. KVX_OCR_UNTRAINED=1 builds it without weights (offline code tests only)."""
    import torch
    from doctr.models import ocr_predictor
    untrained = os.environ.get("KVX_OCR_UNTRAINED") == "1"
    if not untrained:
        use_system_certificates()
    if det_arch is None or reco_arch is None:
        d, r, note = resolve_archs()
        det_arch, reco_arch = det_arch or d, reco_arch or r
        if note:
            print(f"[ocr] {note}")
    try:
        model = ocr_predictor(det_arch=det_arch, reco_arch=reco_arch, pretrained=not untrained,
                              pretrained_backbone=not untrained, assume_straight_pages=True,
                              preserve_aspect_ratio=True, symmetric_pad=True)
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"docTR could not load its weights ({type(e).__name__}: {e}). "
                           f"Run: python scripts\\fetch_ocr_weights.py") from e
    if torch.cuda.is_available():
        model = model.cuda()
    return model


def ocr_available() -> tuple[bool, str]:
    """(ok, message). Actually imports docTR, so a broken install is reported with its real error, together
    with the Python interpreter that is running - the usual cause is starting the app with a Python that is
    not the project's venv (e.g. Anaconda in a new window)."""
    import sys
    try:
        import importlib
        importlib.import_module("doctr")
    except Exception as e:  # noqa: BLE001
        return False, (f"docTR cannot be imported by {sys.executable} ({type(e).__name__}: {e}). "
                       f"Start the app with the project venv's Python, or run: {sys.executable} -m pip install python-doctr")
    try:
        d, r, note = resolve_archs()
    except Exception as e:  # noqa: BLE001
        return False, f"docTR registry unreadable: {e}"
    return True, f"docTR {d} + {r}" + (f" ({note})" if note else "")


def render_pdf(pdf_path, out_dir, scale: float = PDF_SCALE, max_pages: int | None = None) -> list[Path]:
    """Render PDF pages to PNG files (cached: existing files are reused)."""
    import pypdfium2 as pdfium
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = pdfium.PdfDocument(str(pdf_path))
    n = len(pdf) if max_pages is None else min(len(pdf), max_pages)
    paths = []
    for i in range(n):
        p = out_dir / f"{Path(pdf_path).stem}_p{i:03d}.png"
        if not p.exists():
            pdf[i].render(scale=scale).to_pil().convert("RGB").save(p)
        paths.append(p)
    pdf.close()
    return paths


def _page_to_words(page, width: int, height: int, min_conf: float):
    words, lines = [], []
    for block in page.blocks:
        for line in block.lines:
            ids = []
            for w in line.words:
                text = (w.value or "").strip()
                if not text or w.confidence < min_conf:
                    continue
                (x0, y0), (x1, y1) = w.geometry
                ids.append(len(words))
                words.append({"text": text, "box": [float(x0) * width, float(y0) * height,
                                                    float(x1) * width, float(y1) * height],
                              "conf": round(float(w.confidence), 4)})
            if ids:
                lines.append(ids)
    return words, lines


def ocr_images(image_paths: list, det_arch=None, reco_arch=None, min_conf: float = 0.0,
               batch: int = 4) -> list[dict]:
    """-> one dict per image: {"width", "height", "words": [...], "lines": [[word ids], ...]}"""
    model = get_predictor(det_arch, reco_arch)
    out = []
    for i in range(0, len(image_paths), batch):
        ims = [Image.open(p).convert("RGB") for p in image_paths[i:i + batch]]
        result = model([np.asarray(im) for im in ims])
        for im, page in zip(ims, result.pages):
            words, lines = _page_to_words(page, im.width, im.height, min_conf)
            out.append({"width": im.width, "height": im.height, "words": words, "lines": lines})
    return out


def ocr_document(path, doc_id: str = "upload", work_dir=None, max_pages: int = 10,
                 det_arch=None, reco_arch=None) -> list[dict]:
    """Image or PDF -> list of unified page documents (no entities), one per page."""
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        pages = render_pdf(path, Path(work_dir or path.parent) / (path.stem + "_pages"), max_pages=max_pages)
    elif path.suffix.lower() in IMG_EXT:
        pages = [path]
    else:
        raise ValueError(f"unsupported file type {path.suffix} (use an image or a PDF)")
    docs = []
    for k, (img, res) in enumerate(zip(pages, ocr_images(pages, det_arch, reco_arch))):
        docs.append({"doc_id": doc_id if len(pages) == 1 else f"{doc_id}#p{k + 1}", "source": "upload",
                     "split_hint": "upload", "lang": "en", "template_id": None,
                     "image_path": str(Path(img).resolve()), "width": res["width"], "height": res["height"],
                     "words": res["words"], "lines": res["lines"], "entities": [], "links": [],
                     "header_links": [], "annotation": "none"})
    return docs
