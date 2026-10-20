"""Recognition accuracy of PaddleOCR-VL (stock or with a LoRA adapter) on ground-truth line crops.

Same crops and the same text comparison as tools/evaluate.py, so the numbers line up with the
PP-OCRv6 "Recognition accuracy" and "Character error rate" rows.

    python vl/evaluate.py --name vl_stock
    python vl/evaluate.py --name vl_lora --adapter output/vl_lora
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from rapidfuzz.distance import Levenshtein
from transformers import AutoModelForImageTextToText, AutoProcessor

from data import MODEL, generation_batch, load_lines


def norm(text: str) -> str:
    return "".join(text.split())


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True)
    p.add_argument("--adapter", help="LoRA adapter folder from vl/train_lora.py")
    p.add_argument("--model", default=MODEL)
    p.add_argument("--data", default="data/cord")
    p.add_argument("--split", default="test")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--limit", type=int)
    p.add_argument("--out-dir", default="results/cord")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    processor = AutoProcessor.from_pretrained(args.model)
    processor.tokenizer.padding_side = "left"
    model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=dtype).to(device).eval()
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()

    pairs = load_lines(args.data, args.split, args.limit)
    preds, start = [], time.time()
    for i in range(0, len(pairs), args.batch_size):
        chunk = pairs[i:i + args.batch_size]
        batch = generation_batch(processor, [path for path, _ in chunk]).to(device)
        with torch.no_grad():
            out = model.generate(**batch, max_new_tokens=64, do_sample=False)
        new = out[:, batch["input_ids"].shape[1]:]
        preds += processor.batch_decode(new, skip_special_tokens=True)
    elapsed = time.time() - start

    labels = [norm(t) for _, t in pairs]
    preds = [norm(t) for t in preds]
    result = {
        "name": args.name, "model": args.model, "adapter": args.adapter, "split": args.split,
        "recognition": {
            "accuracy": sum(a == b for a, b in zip(preds, labels)) / len(labels),
            "cer": sum(Levenshtein.distance(a, b) for a, b in zip(preds, labels)) / sum(len(b) for b in labels),
            "lines": len(labels),
            "ms_per_line": 1000 * elapsed / len(labels),
        },
        "device": torch.cuda.get_device_name() if device == "cuda" else "cpu",
    }
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.name}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
