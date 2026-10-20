"""LoRA fine-tuning of PaddleOCR-VL-1.6 for text-line recognition, on one 16 GB GPU (e.g. a free Colab T4).

Uses the line crops written by tools/prepare_data.py, so it trains on exactly the data the
PP-OCRv6 recognizer is fine-tuned on:

    pip install -r vl/requirements.txt
    python vl/train_lora.py --data data/cord --out output/vl_lora

The vision encoder is frozen; LoRA adapters go on the language model's attention and MLP layers.
Weights stay in float32 and the forward pass runs in float16 autocast, because the T4 has no bfloat16.
"""

from __future__ import annotations

import argparse
import math
import random
import time

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForImageTextToText, AutoProcessor, get_cosine_schedule_with_warmup

from data import MODEL, load_lines, training_batch


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data/cord")
    p.add_argument("--out", default="output/vl_lora")
    p.add_argument("--model", default=MODEL)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--limit", type=int, help="use only the first N training lines (smoke tests)")
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = AutoProcessor.from_pretrained(args.model)
    processor.tokenizer.padding_side = "right"
    model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.float32).to(device)
    model = get_peft_model(model, LoraConfig(
        r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05,
        target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)",
    ))
    model.print_trainable_parameters()

    pairs = load_lines(args.data, "train", args.limit)
    steps_per_epoch = math.ceil(len(pairs) / args.batch_size)
    total = int(steps_per_epoch * args.epochs)
    optimizer = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad], lr=args.lr, weight_decay=0.0)
    scheduler = get_cosine_schedule_with_warmup(optimizer, max(1, total // 20), total)
    scaler = torch.amp.GradScaler(enabled=device == "cuda")

    model.train()
    step, start, running = 0, time.time(), 0.0
    while step < total:
        random.shuffle(pairs)
        for i in range(0, len(pairs), args.batch_size):
            if step >= total:
                break
            batch = {k: v.to(device) for k, v in training_batch(processor, pairs[i:i + args.batch_size]).items()}
            with torch.autocast(device, dtype=torch.float16, enabled=device == "cuda"):
                loss = model(**batch).loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            step += 1
            running += loss.item()
            if step % args.log_every == 0 or step == total:
                n = args.log_every if step % args.log_every == 0 else step % args.log_every
                print(f"step {step}/{total}  loss {running / n:.4f}  lr {scheduler.get_last_lr()[0]:.2e}  "
                      f"{(time.time() - start) / step:.2f}s/step", flush=True)
                running = 0.0

    model.save_pretrained(args.out)
    processor.save_pretrained(args.out)
    print(f"saved LoRA adapter to {args.out}")


if __name__ == "__main__":
    main()
