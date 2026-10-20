"""Draw two evaluation runs side by side on the same images.

Green box: text matches the label exactly. Red box: wrong text, or no matching label.

    python tools/visualize.py results/cord/stock_predictions.json results/cord/finetuned_predictions.json \
        --data data/cord --out docs/assets/comparison.jpg --images 3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from evaluate import IGNORE, iou, norm, polygon

GREEN, RED = (22, 163, 74), (220, 38, 38)


def load_font(size: int):
    for path in ("/System/Library/Fonts/Supplemental/Arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                 "C:/Windows/Fonts/arial.ttf"):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size)


def labels(data: Path, split: str) -> dict:
    out = {}
    for line in (data / f"det_{split}.txt").read_text(encoding="utf-8").splitlines():
        rel, ann = line.split("\t", 1)
        out[rel] = [g for g in json.loads(ann) if g["transcription"] != IGNORE]
    return out


def correct(pred: dict, gts: list) -> bool:
    poly = polygon(pred["points"])
    best = max(gts, key=lambda g: iou(poly, polygon(g["points"])), default=None)
    return best is not None and iou(poly, polygon(best["points"])) >= 0.5 and \
        norm(pred["text"]) == norm(best["transcription"])


def panel(image: Image.Image, preds: list, gts: list, title: str, font, title_font) -> tuple[Image.Image, int]:
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    right = 0
    for p in preds:
        ok = correct(p, gts)
        right += ok
        pts = [tuple(map(int, xy)) for xy in p["points"]]
        draw.polygon(pts, outline=GREEN if ok else RED, width=3)
        if not ok:
            x, y = min(x for x, _ in pts), max(y for _, y in pts)
            draw.text((x, y + 1), p["text"], fill=RED, font=font, stroke_width=3, stroke_fill="white")
    header = Image.new("RGB", (canvas.width, title_font.size + 24), "white")
    ImageDraw.Draw(header).text((12, 10), f"{title}: {right}/{len(gts)} lines correct", fill="black",
                                font=title_font)
    out = Image.new("RGB", (canvas.width, canvas.height + header.height), "white")
    out.paste(header, (0, 0))
    out.paste(canvas, (0, header.height))
    return out, right


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("before")
    p.add_argument("after")
    p.add_argument("--data", default="data/cord")
    p.add_argument("--split", default="test")
    p.add_argument("--out", default="comparison.jpg")
    p.add_argument("--images", type=int, default=3, help="how many images (largest improvement first)")
    p.add_argument("--height", type=int, default=900)
    args = p.parse_args()

    data = Path(args.data)
    gt = labels(data, args.split)
    before = {e["image"]: e["predictions"] for e in json.loads(Path(args.before).read_text())}
    after = {e["image"]: e["predictions"] for e in json.loads(Path(args.after).read_text())}
    names = [Path(args.before).stem.replace("_predictions", ""), Path(args.after).stem.replace("_predictions", "")]

    def gain(rel):
        return sum(correct(q, gt[rel]) for q in after[rel]) - sum(correct(q, gt[rel]) for q in before[rel])

    chosen = sorted(before, key=gain, reverse=True)[: args.images]
    rows = []
    for rel in chosen:
        image = Image.open(data / rel)
        scale = args.height / image.height
        font, title_font = load_font(max(12, int(16 / scale))), load_font(max(16, int(26 / scale)))
        left, _ = panel(image, before[rel], gt[rel], names[0], font, title_font)
        right, _ = panel(image, after[rel], gt[rel], names[1], font, title_font)
        pair = np.hstack([np.asarray(left), np.full((left.height, 12, 3), 255, np.uint8), np.asarray(right)])
        pair = cv2.resize(pair, (int(pair.shape[1] * scale), args.height), interpolation=cv2.INTER_AREA)
        rows.append(pair)
    width = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((0, 24), (0, width - r.shape[1]), (0, 0)), constant_values=255) for r in rows]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.vstack(rows)).save(args.out, quality=90)
    print(f"saved {args.out}: {', '.join(chosen)}")


if __name__ == "__main__":
    main()
