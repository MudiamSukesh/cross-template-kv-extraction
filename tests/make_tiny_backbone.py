"""Build a tiny, randomly initialised LayoutLMv3 (+ BPE tokenizer + image processor) saved in
the same on-disk format as microsoft/layoutlmv3-base. Used only to test code paths offline;
numbers from it mean nothing."""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--corpus", help="jsonl of docs to train the BPE on")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from tokenizers import ByteLevelBPETokenizer
    texts = ["Name Date Address Phone Account Amount Signature City PIN Code Email form page orphan"]
    if args.corpus:
        for line in open(args.corpus, encoding="utf-8"):
            d = json.loads(line)
            texts.append(" ".join(w["text"] for w in d["words"]))
    bpe = ByteLevelBPETokenizer()
    bpe.train_from_iterator(texts * 3, vocab_size=600, min_frequency=1,
                            special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"])
    bpe.save_model(str(out))

    try:
        from transformers import LayoutLMv3TokenizerFast as Tok
    except ImportError:  # transformers 5 renamed the fast class
        from transformers import LayoutLMv3Tokenizer as Tok
    vocab = json.loads((out / "vocab.json").read_text(encoding="utf-8"))
    merges = [tuple(l.split(" ")) for l in (out / "merges.txt").read_text(encoding="utf-8").splitlines()
              if l and not l.startswith("#version")]
    tok = None
    try:  # transformers 5: vocab / merges objects
        tok = Tok(vocab=vocab, merges=merges, add_prefix_space=True)
    except Exception:  # noqa: BLE001 - transformers 4 rejects these arguments in several ways
        tok = None
    if tok is None or len(tok) < 50:  # transformers 4: file paths
        tok = Tok(vocab_file=str(out / "vocab.json"), merges_file=str(out / "merges.txt"), add_prefix_space=True)
    assert len(tok) >= 50, f"tokenizer vocabulary not loaded ({len(tok)})"
    tok.save_pretrained(out)

    from transformers import LayoutLMv3Config, LayoutLMv3ImageProcessor, LayoutLMv3Model
    cfg = LayoutLMv3Config(vocab_size=len(tok), hidden_size=64, num_hidden_layers=2, num_attention_heads=2,
                           intermediate_size=128, coordinate_size=8, shape_size=16, max_position_embeddings=514,
                           input_size=224, patch_size=16, pad_token_id=tok.pad_token_id,
                           bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id)
    LayoutLMv3Model(cfg).save_pretrained(out)
    LayoutLMv3ImageProcessor(apply_ocr=False).save_pretrained(out)
    print("tiny backbone written to", out, "| vocab", len(tok))


if __name__ == "__main__":
    main()
