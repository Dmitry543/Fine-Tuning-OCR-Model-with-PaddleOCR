"""Shared data handling for PaddleOCR-VL line recognition (prompt "OCR:")."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

MODEL = "PaddlePaddle/PaddleOCR-VL-1.6"
PROMPT = "OCR:"


def load_lines(data: str, split: str, limit: int | None = None) -> list[tuple[str, str]]:
    """(image path, text) pairs from the rec_{split}.txt written by tools/prepare_data.py."""
    root = Path(data)
    pairs = []
    for line in (root / f"rec_{split}.txt").read_text(encoding="utf-8").splitlines():
        rel, text = line.split("\t", 1)
        pairs.append((str(root / rel), text))
    return pairs[:limit] if limit else pairs


def conversation(image: Image.Image, answer: str | None = None) -> list[dict]:
    msgs = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": PROMPT}]}]
    if answer is not None:
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
    return msgs


def training_batch(processor, pairs: list[tuple[str, str]]) -> dict:
    """Tokenise image+prompt+answer; only the answer and its end token are trained on."""
    images = [Image.open(p).convert("RGB") for p, _ in pairs]
    batch = processor.apply_chat_template([conversation(im, t) for im, (_, t) in zip(images, pairs)],
                                          tokenize=True, return_dict=True, return_tensors="pt", padding=True)
    labels = torch.full_like(batch["input_ids"], -100)
    lengths = batch["attention_mask"].sum(1)
    for i, (_, text) in enumerate(pairs):
        n = len(processor.tokenizer(text, add_special_tokens=False)["input_ids"]) + 1  # + </s>
        end = int(lengths[i])  # right padding: real tokens are [0, end)
        labels[i, end - n:end] = batch["input_ids"][i, end - n:end]
    batch["labels"] = labels
    return batch


def generation_batch(processor, paths: list[str]) -> dict:
    images = [Image.open(p).convert("RGB") for p in paths]
    return processor.apply_chat_template([conversation(im) for im in images], add_generation_prompt=True,
                                         tokenize=True, return_dict=True, return_tensors="pt", padding=True)
