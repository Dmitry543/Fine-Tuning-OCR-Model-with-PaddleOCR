"""Gradio demo: stock PP-OCRv6 vs the fine-tuned models on the same image.

    pip install gradio
    python demo/app.py --det output/inference/det --rec output/inference/rec
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gradio as gr
import numpy as np
from paddleocr import PaddleOCR
from PIL import Image, ImageDraw

COMMON = dict(use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False)


def build(det: str | None, rec: str | None) -> PaddleOCR:
    kw = dict(text_detection_model_name="PP-OCRv6_small_det", text_recognition_model_name="PP-OCRv6_small_rec")
    if det:
        kw["text_detection_model_dir"] = det
    if rec:
        kw["text_recognition_model_dir"] = rec
    return PaddleOCR(**kw, **COMMON)


def run(ocr: PaddleOCR, image: np.ndarray):
    res = next(iter(ocr.predict(image[:, :, ::-1].copy())))  # Gradio gives RGB, PaddleOCR expects BGR
    canvas = Image.fromarray(image).convert("RGB")
    draw = ImageDraw.Draw(canvas)
    for poly in res["rec_polys"]:
        draw.polygon([tuple(map(int, p)) for p in poly], outline=(37, 99, 235), width=3)
    return canvas, "\n".join(res["rec_texts"])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--det", default="output/inference/det")
    p.add_argument("--rec", default="output/inference/rec")
    p.add_argument("--examples", default="demo/examples")
    p.add_argument("--share", action="store_true")
    args = p.parse_args()

    stock, tuned = build(None, None), build(args.det, args.rec)

    def compare(image):
        if image is None:
            return None, "", None, ""
        return (*run(stock, image), *run(tuned, image))

    with gr.Blocks(title="PP-OCRv6: stock vs fine-tuned") as app:
        gr.Markdown("## PP-OCRv6: stock vs fine-tuned\nUpload a receipt or document photo.")
        image = gr.Image(type="numpy", label="Input")
        with gr.Row():
            with gr.Column():
                stock_img, stock_txt = gr.Image(label="Stock PP-OCRv6_small"), gr.Textbox(label="Text", lines=12)
            with gr.Column():
                tuned_img, tuned_txt = gr.Image(label="Fine-tuned"), gr.Textbox(label="Text", lines=12)
        examples = sorted(str(p) for p in Path(args.examples).glob("*.jpg"))
        if examples:
            gr.Examples(examples, inputs=image)
        image.change(compare, image, [stock_img, stock_txt, tuned_img, tuned_txt])
    app.launch(share=args.share)


if __name__ == "__main__":
    main()
