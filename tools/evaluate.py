"""Evaluate an OCR model pair on a prepared dataset and compare runs.

Three metrics, all computed with the same PaddleOCR 3.x inference code people deploy with:

    detection     precision / recall / F1 of text-line boxes (IoU >= 0.5, one-to-one)
    recognition   exact-match accuracy and character error rate on ground-truth line crops
    end-to-end    F1 where a prediction counts only if its box matches AND its text is exact

Text comparison ignores whitespace; case and punctuation count.

Usage:
    # stock PP-OCRv6 models (downloaded automatically)
    python tools/evaluate.py --name stock --det PP-OCRv6_small_det --rec PP-OCRv6_small_rec

    # your exported models
    python tools/evaluate.py --name finetuned --det output/inference/det --rec output/inference/rec

    # side-by-side table
    python tools/evaluate.py --compare results/stock.json results/finetuned.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from rapidfuzz.distance import Levenshtein
from shapely.geometry import Polygon

IGNORE = "###"


def model_kwargs(kind: str, value: str) -> dict:
    """A path to an exported model, or the name of an official one."""
    key = "text_detection" if kind == "det" else "text_recognition"
    if Path(value).exists():
        return {f"{key}_model_dir": value, f"{key}_model_name": read_model_name(value)}
    return {f"{key}_model_name": value}


def read_model_name(model_dir: str) -> str:
    import yaml

    cfg = yaml.safe_load(open(Path(model_dir) / "inference.yml"))
    return cfg["Global"]["model_name"]


def polygon(points) -> Polygon:
    p = Polygon(np.asarray(points, float))
    return p if p.is_valid else p.buffer(0)


def iou(a: Polygon, b: Polygon) -> float:
    inter = a.intersection(b).area
    return inter / (a.area + b.area - inter + 1e-9)


def norm(text: str) -> str:
    """Compare text with whitespace removed (PaddleOCR's RecMetric default): datasets disagree on
    spacing conventions such as "( L )" vs "(L)". Case and punctuation still count."""
    return "".join(text.split())


def match(preds, gts, threshold=0.5):
    """Greedy one-to-one matching by IoU. Predictions mostly inside an ignore region are dropped."""
    care = [g for g in gts if g["transcription"] != IGNORE]
    ignore = [polygon(g["points"]) for g in gts if g["transcription"] == IGNORE]
    kept = []
    for p in preds:
        poly = polygon(p["points"])
        if poly.area > 0 and not any(poly.intersection(i).area / poly.area > 0.5 for i in ignore):
            kept.append((poly, p["text"]))
    gt_polys = [polygon(g["points"]) for g in care]
    pairs = sorted(((iou(pp, gp), i, j) for i, (pp, _) in enumerate(kept) for j, gp in enumerate(gt_polys)),
                   reverse=True)
    used_p, used_g, matched = set(), set(), []
    for score, i, j in pairs:
        if score < threshold:
            break
        if i not in used_p and j not in used_g:
            used_p.add(i)
            used_g.add(j)
            matched.append((kept[i][1], care[j]["transcription"]))
    return len(kept), len(care), matched


def prf(tp: int, n_pred: int, n_gt: int) -> dict:
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_gt if n_gt else 0.0
    return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0}


def evaluate(args) -> dict:
    from paddleocr import PaddleOCR, TextRecognition

    data = Path(args.data)
    det_lines = (data / f"det_{args.split}.txt").read_text(encoding="utf-8").splitlines()
    rec_lines = (data / f"rec_{args.split}.txt").read_text(encoding="utf-8").splitlines()
    device = args.device

    # recognition on ground-truth crops
    rec_kw = {k.replace("text_recognition_", ""): v for k, v in model_kwargs("rec", args.rec).items()}
    engine = {"engine": args.engine} if args.engine else {}
    engine["enable_mkldnn"] = False  # PaddlePaddle 3.3.1's oneDNN executor fails on PP-OCRv6 on x86 CPUs
    recognizer = TextRecognition(device=device, **engine, **rec_kw)
    crops = [str(data / line.split("\t")[0]) for line in rec_lines]
    labels = [norm(line.split("\t", 1)[1]) for line in rec_lines]
    preds, start = [], time.time()
    for res in recognizer.predict(crops, batch_size=args.batch_size):
        preds.append(norm(res["rec_text"]))
    rec_time = time.time() - start
    correct = sum(p == g for p, g in zip(preds, labels))
    edits = sum(Levenshtein.distance(p, g) for p, g in zip(preds, labels))
    recognition = {"accuracy": correct / len(labels), "cer": edits / sum(len(g) for g in labels),
                   "lines": len(labels), "ms_per_line": 1000 * rec_time / len(labels)}

    # full pipeline on images
    ocr = PaddleOCR(device=device, use_doc_orientation_classify=False, use_doc_unwarping=False,
                    use_textline_orientation=False, **engine,
                    **model_kwargs("det", args.det), **model_kwargs("rec", args.rec))
    n_pred = n_gt = det_tp = e2e_tp = 0
    examples, start = [], time.time()
    for line in det_lines:
        rel, ann = line.split("\t", 1)
        gts = json.loads(ann)
        res = next(iter(ocr.predict(str(data / rel))))
        preds = [{"points": np.asarray(b).tolist(), "text": t} for b, t in zip(res["rec_polys"], res["rec_texts"])]
        np_, ng, matched = match(preds, gts)
        n_pred, n_gt, det_tp = n_pred + np_, n_gt + ng, det_tp + len(matched)
        e2e_tp += sum(norm(p) == norm(g) for p, g in matched)
        examples.append({"image": rel, "predictions": preds})
    pipe_time = time.time() - start

    result = {
        "name": args.name, "det": args.det, "rec": args.rec, "split": args.split, "images": len(det_lines),
        "detection": prf(det_tp, n_pred, n_gt),
        "recognition": recognition,
        "end_to_end": prf(e2e_tp, n_pred, n_gt),
        "s_per_image": pipe_time / len(det_lines),
        "device": device,
        "engine": args.engine or "paddle",
    }
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.name}.json").write_text(json.dumps(result, indent=2))
    (out / f"{args.name}_predictions.json").write_text(json.dumps(examples, ensure_ascii=False))
    print(json.dumps(result, indent=2))
    return result


def compare(paths: list[str]):
    runs = [json.loads(Path(p).read_text()) for p in paths]
    rows = [
        ("Detection F1", lambda r: r["detection"]["f1"], True),
        ("Detection precision", lambda r: r["detection"]["precision"], True),
        ("Detection recall", lambda r: r["detection"]["recall"], True),
        ("Recognition accuracy", lambda r: r["recognition"]["accuracy"], True),
        ("Character error rate", lambda r: r["recognition"]["cer"], False),
        ("End-to-end F1", lambda r: r["end_to_end"]["f1"], True),
    ]
    print("| Metric | " + " | ".join(r["name"] for r in runs) + (" | Change |" if len(runs) == 2 else " |"))
    print("|---" * (len(runs) + 1 + (len(runs) == 2)) + "|")
    for label, get, higher in rows:
        vals = [get(r) for r in runs]
        cells = " | ".join(f"{100 * v:.1f}%" for v in vals)
        if len(runs) == 2:
            delta = 100 * (vals[1] - vals[0])
            cells += f" | {delta:+.1f} pt"
        print(f"| {label} | {cells} |")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", help="run name, used for the output file")
    p.add_argument("--det", help="detection model: exported folder or official model name")
    p.add_argument("--rec", help="recognition model: exported folder or official model name")
    p.add_argument("--data", default="data/cord")
    p.add_argument("--split", default="test")
    p.add_argument("--device", default="cpu", help="cpu or gpu:0")
    p.add_argument("--engine", help="inference engine, e.g. onnxruntime for models from tools/export_onnx.py")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--out-dir", default="results")
    p.add_argument("--compare", nargs="+", metavar="RESULT_JSON")
    args = p.parse_args()
    if args.compare:
        compare(args.compare)
    elif args.name and args.det and args.rec:
        evaluate(args)
    else:
        p.error("give --name, --det and --rec, or --compare")


if __name__ == "__main__":
    main()
