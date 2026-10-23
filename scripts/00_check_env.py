r"""Step 0 - check the environment and download/verify the LayoutLMv3 backbone.

  python scripts\00_check_env.py
  python scripts\00_check_env.py --backbone D:\models\layoutlmv3-base     (a local copy)
"""
import argparse
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="microsoft/layoutlmv3-base")
    args = ap.parse_args()
    print(f"python        {sys.version.split()[0]}")
    for m in ("torch", "transformers", "tokenizers", "numpy", "scipy", "PIL", "yaml", "flask", "doctr", "pypdfium2"):
        try:
            importlib.import_module(m)
            from importlib.metadata import version
            v = version({"PIL": "pillow", "yaml": "pyyaml", "doctr": "python-doctr"}.get(m, m))
            print(f"{m:13s} {v}")
        except Exception as e:  # noqa: BLE001
            print(f"{m:13s} MISSING ({type(e).__name__}) -> pip install -r requirements.txt")
    import torch
    if torch.cuda.is_available():
        print(f"GPU           {torch.cuda.get_device_name(0)} | bf16={torch.cuda.is_bf16_supported()} "
              f"| CUDA {torch.version.cuda} | {torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GB")
        x = torch.randn(256, 256, device="cuda")
        print(f"GPU matmul    ok ({float((x @ x).sum()):.1f})")
    else:
        print("GPU           none (training on CPU is too slow for the real backbone)")

    print("\nchecking OCR (docTR; downloads its weights the first time) ...")
    try:
        import tempfile
        from PIL import Image, ImageDraw, ImageFont
        from kvx.ocr import ocr_available, ocr_images
        im = Image.new("RGB", (900, 260), "white")
        dr = ImageDraw.Draw(im)
        try:
            font = ImageFont.truetype("arial.ttf", 40)
        except OSError:
            font = ImageFont.load_default()
        dr.text((40, 40), "Name of Registrant: DLA Piper", fill="black", font=font)
        dr.text((40, 140), "Date: 12/03/2026", fill="black", font=font)
        p = Path(tempfile.mkdtemp()) / "ocr_check.png"
        im.save(p)
        res = ocr_images([p])[0]
        print(f"OCR ok - {ocr_available()[1]}: {len(res['words'])} words in {len(res['lines'])} lines -> "
              + " | ".join(" ".join(res["words"][i]["text"] for i in l) for l in res["lines"]))
    except Exception as e:  # noqa: BLE001
        print(f"OCR FAILED: {type(e).__name__}: {e}")

    print(f"\nloading backbone {args.backbone} ...")
    from kvx.runtime import load_image_processor, load_tokenizer
    from transformers import LayoutLMv3Model
    tok = load_tokenizer(args.backbone)
    load_image_processor(args.backbone)
    model = LayoutLMv3Model.from_pretrained(args.backbone)
    enc = tok(["Name:", "Ravi", "Kumar"], boxes=[[10, 10, 60, 30], [70, 10, 120, 30], [125, 10, 190, 30]],
              return_tensors="pt")
    with torch.no_grad():
        out = model(input_ids=enc["input_ids"], bbox=enc["bbox"], attention_mask=enc["attention_mask"])
    print(f"backbone ok: hidden={model.config.hidden_size}, layers={model.config.num_hidden_layers}, "
          f"output {tuple(out.last_hidden_state.shape)}")
    print("\nEnvironment ready.")


if __name__ == "__main__":
    main()
