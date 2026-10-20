"""Convert an OCR dataset into PaddleOCR detection and recognition labels.

Output layout (all paths in label files are relative to OUT):

    OUT/images/{split}/*.jpg       full images
    OUT/det_{split}.txt            <image>\t[{"transcription": str, "points": [[x, y], ...]}, ...]
    OUT/rec/{split}/*.jpg          cropped text lines
    OUT/rec_{split}.txt            <crop>\t<text>

Ignored regions use the transcription "###": detection skips them, recognition never sees them.

Sources:
    cord        CORD-v2 receipts (CC BY 4.0), downloaded from Hugging Face
    ppocrlabel  a folder labelled with PPOCRLabel (Label.txt), split randomly
    totaltext   Total-Text polygon annotations (download it yourself; see docs/datasets.md)
"""

from __future__ import annotations

import argparse
import ast
import io
import json
import random
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

IGNORE = "###"
CORD_REPO = "https://huggingface.co/datasets/naver-clova-ix/cord-v2/resolve/main/data"


# ---------------------------------------------------------------- writing


class Writer:
    def __init__(self, out: Path, max_side: int):
        self.out = out
        self.max_side = max_side
        self.files: dict[str, object] = {}

    def __enter__(self):
        self.out.mkdir(parents=True, exist_ok=True)
        return self

    def __exit__(self, *exc):
        for f in self.files.values():
            f.close()

    def _file(self, name: str):
        if name not in self.files:
            self.files[name] = open(self.out / name, "w", encoding="utf-8")
        return self.files[name]

    def add(self, split: str, name: str, image: np.ndarray, regions: list[tuple[str, np.ndarray]]):
        """Write one image with its (text, polygon) regions to both label sets."""
        scale = min(1.0, self.max_side / max(image.shape[:2]))
        if scale < 1.0:
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        regions = [(t, p * scale) for t, p in regions]

        rel = f"images/{split}/{name}.jpg"
        (self.out / rel).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(self.out / rel), image, [cv2.IMWRITE_JPEG_QUALITY, 95])

        det = [{"transcription": t, "points": np.round(p).astype(int).tolist()} for t, p in regions]
        self._file(f"det_{split}.txt").write(f"{rel}\t{json.dumps(det, ensure_ascii=False)}\n")

        for i, (text, poly) in enumerate(regions):
            if text == IGNORE or not text.strip():
                continue
            crop = crop_text(image, poly)
            if crop is None:
                continue
            crop_rel = f"rec/{split}/{name}_{i:03d}.jpg"
            (self.out / crop_rel).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(self.out / crop_rel), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            self._file(f"rec_{split}.txt").write(f"{crop_rel}\t{text}\n")


def crop_text(image: np.ndarray, poly: np.ndarray) -> np.ndarray | None:
    """Rectify the minimum-area rectangle around a polygon, like PaddleOCR does at inference."""
    rect = cv2.minAreaRect(poly.astype(np.float32))
    box = cv2.boxPoints(rect)
    # order: top-left, top-right, bottom-right, bottom-left
    box = box[np.argsort(box[:, 0])]
    left, right = box[:2], box[2:]
    tl, bl = left[np.argsort(left[:, 1])]
    tr, br = right[np.argsort(right[:, 1])]
    src = np.array([tl, tr, br, bl], dtype=np.float32)
    w = int(round(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))))
    h = int(round(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))))
    if w < 4 or h < 4:
        return None
    dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    crop = cv2.warpPerspective(image, cv2.getPerspectiveTransform(src, dst), (w, h),
                               borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_CUBIC)
    if h / w >= 1.5:  # vertical line: rotate so text reads left to right
        crop = np.rot90(crop)
    return crop


# ---------------------------------------------------------------- CORD


def quad(q: dict) -> np.ndarray:
    return np.array([[q["x1"], q["y1"]], [q["x2"], q["y2"]], [q["x3"], q["y3"]], [q["x4"], q["y4"]]], float)


def cord_lines(gt: dict, gap: float) -> list[tuple[str, np.ndarray]]:
    """Merge CORD word boxes into text lines, the unit PP-OCR detects and recognises.

    Words sharing a row_id are joined left to right; a horizontal gap wider than
    `gap` x the word height starts a new line (receipt columns are separate lines).
    """
    rows: dict[int, list[dict]] = {}
    for line in gt["valid_line"]:
        for w in line["words"]:
            if w["text"].strip():
                rows.setdefault(w["row_id"], []).append(w)

    regions = []
    for words in rows.values():
        words.sort(key=lambda w: quad(w["quad"])[:, 0].min())
        segment = [words[0]]
        for w in words[1:]:
            prev, cur = quad(segment[-1]["quad"]), quad(w["quad"])
            height = max(np.ptp(prev[:, 1]), np.ptp(cur[:, 1]), 1)
            if cur[:, 0].min() - prev[:, 0].max() > gap * height:
                regions.append(merge(segment))
                segment = []
            segment.append(w)
        regions.append(merge(segment))

    for group in gt.get("repeating_symbol", []) + gt.get("dontcare", []):
        for item in group:
            regions.append((IGNORE, quad(item.get("quad", item))))
    return regions


def merge(words: list[dict]) -> tuple[str, np.ndarray]:
    first, last = quad(words[0]["quad"]), quad(words[-1]["quad"])
    poly = np.array([first[0], last[1], last[2], first[3]])
    return " ".join(w["text"].strip() for w in words), poly


def prepare_cord(args, writer: Writer):
    import pyarrow.parquet as pq
    import requests

    raw = Path(args.raw)
    raw.mkdir(parents=True, exist_ok=True)
    api = "https://huggingface.co/api/datasets/naver-clova-ix/cord-v2/tree/main/data"
    for entry in requests.get(api, timeout=60).json():
        dest = raw / Path(entry["path"]).name
        if dest.exists() and dest.stat().st_size == entry["size"]:
            continue
        print(f"downloading {dest.name} ({entry['size'] / 1e6:.0f} MB)")
        with requests.get(f"{CORD_REPO}/{dest.name}", stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)

    splits = {"train": "train", "validation": "val", "test": "test"}
    for prefix, split in splits.items():
        for path in sorted(raw.glob(f"{prefix}-*.parquet")):
            table = pq.read_table(path)
            for row in tqdm(table.to_pylist(), desc=f"cord {split} {path.name[:20]}"):
                gt = json.loads(row["ground_truth"])
                image = cv2.cvtColor(np.array(Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")),
                                     cv2.COLOR_RGB2BGR)
                name = f"{split}_{gt['meta']['image_id']:04d}"
                image, regions = crop_to_labels(image, cord_lines(gt, args.line_gap))
                writer.add(split, name, image, regions)


def crop_to_labels(image: np.ndarray, regions: list[tuple[str, np.ndarray]]):
    """CORD labels only the menu/total block; headers and footers are blurred or unlabelled.
    Cropping to the labelled block keeps unlabelled text from counting as background
    in training and as false positives in evaluation."""
    points = np.concatenate([p for _, p in regions])
    heights = [np.ptp(p[:, 1]) for t, p in regions if t != IGNORE] or [20]
    pad = 2 * float(np.median(heights))
    h, w = image.shape[:2]
    x0, y0 = (np.maximum(points.min(0) - pad, 0)).astype(int)
    x1, y1 = (np.minimum(points.max(0) + pad, [w, h])).astype(int)
    return image[y0:y1, x0:x1], [(t, p - [x0, y0]) for t, p in regions]


# ---------------------------------------------------------------- PPOCRLabel


def prepare_ppocrlabel(args, writer: Writer):
    root = Path(args.input)
    label_file = root / "Label.txt"
    entries = []
    for line in label_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rel, ann = line.split("\t", 1)
        # PPOCRLabel stores paths as "<folder>/<image>", relative to the folder's parent
        entries.append((root.parent / rel, json.loads(ann)))

    random.Random(args.seed).shuffle(entries)
    n_test = max(1, int(len(entries) * args.test_ratio))
    n_val = max(1, int(len(entries) * args.val_ratio))
    parts = {"test": entries[:n_test], "val": entries[n_test:n_test + n_val], "train": entries[n_test + n_val:]}

    for split, items in parts.items():
        for path, ann in tqdm(items, desc=f"ppocrlabel {split}"):
            image = cv2.imread(str(path))
            if image is None:
                print(f"skip unreadable image: {path}")
                continue
            regions = [(a["transcription"] if not a.get("difficult") else IGNORE, np.array(a["points"], float))
                       for a in ann]
            writer.add(split, path.stem, image, regions)


# ---------------------------------------------------------------- Total-Text


TT_LINE = re.compile(r"x: \[\[(?P<x>[^\]]*)\]\], y: \[\[(?P<y>[^\]]*)\]\], ornt: \[(?P<o>[^\]]*)\], "
                     r"transcriptions: \[(?P<t>.*)\]$")


def prepare_totaltext(args, writer: Writer):
    root = Path(args.input)
    ann_root = root / "Annotation" / "groundtruth_polygonal_annotation"
    for src_split, split in (("Train", "train"), ("Test", "test")):
        for img_path in tqdm(sorted((root / src_split).iterdir()), desc=f"totaltext {split}"):
            ann = ann_root / src_split / f"poly_gt_{img_path.stem}.txt"
            image = cv2.imread(str(img_path))
            if image is None or not ann.exists():
                print(f"skip {img_path.name}: missing image or annotation")
                continue
            # an annotation can wrap across lines; every record starts with "x:"
            records = re.split(r"\n(?=x:)", ann.read_text(encoding="utf-8").strip())
            regions = []
            for rec in records:
                m = TT_LINE.match(" ".join(rec.split()).replace("[ ", "["))
                if not m:
                    print(f"skip unparsable record in {ann.name}: {rec[:60]!r}")
                    continue
                xs = [int(v) for v in m["x"].split()]
                ys = [int(v) for v in m["y"].split()]
                text = ast.literal_eval(m["t"].replace("u'", "'").replace('u"', '"'))
                ignore = ast.literal_eval(m["o"].replace("u'", "'")) == "#" or text == "#"
                regions.append((IGNORE if ignore else text, np.array(list(zip(xs, ys)), float)))
            writer.add(split, img_path.stem, image, regions)


# ---------------------------------------------------------------- CLI


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("source", choices=["cord", "ppocrlabel", "totaltext"])
    p.add_argument("--out", required=True, help="output dataset folder, e.g. data/cord")
    p.add_argument("--input", help="dataset folder (ppocrlabel, totaltext)")
    p.add_argument("--raw", default="data/raw/cord", help="CORD download cache")
    p.add_argument("--max-side", type=int, default=1600, help="downscale images whose longest side is larger")
    p.add_argument("--line-gap", type=float, default=1.0, help="CORD: word gap (x height) that splits a line")
    p.add_argument("--val-ratio", type=float, default=0.1, help="PPOCRLabel: share of images for validation")
    p.add_argument("--test-ratio", type=float, default=0.1, help="PPOCRLabel: share of images for testing")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty; delete it or choose another --out")
    if args.source != "cord" and not args.input:
        raise SystemExit(f"--input is required for {args.source}")

    with Writer(out, args.max_side) as writer:
        {"cord": prepare_cord, "ppocrlabel": prepare_ppocrlabel, "totaltext": prepare_totaltext}[args.source](
            args, writer)

    for f in sorted(out.glob("*.txt")):
        print(f"{f}: {sum(1 for _ in open(f, encoding='utf-8'))} lines")


if __name__ == "__main__":
    main()
