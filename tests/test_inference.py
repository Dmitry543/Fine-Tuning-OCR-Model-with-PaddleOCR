"""Run the released models through PaddleOCR on CPU. Skipped unless paddleocr is installed (see CI)."""

import io
import os
import urllib.request
import zipfile
from pathlib import Path

import pytest

paddleocr = pytest.importorskip("paddleocr")
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

ROOT = Path(__file__).resolve().parents[1]
RELEASE = "https://github.com/Dmitry543/Fine-Tuning-OCR-Model-with-PaddleOCR/releases/download/v2.0.0/pp-ocrv6-small-cord.zip"
COMMON = dict(use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False,
              text_detection_model_name="PP-OCRv6_small_det", text_recognition_model_name="PP-OCRv6_small_rec",
              enable_mkldnn=False)


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    out = tmp_path_factory.mktemp("models")
    with urllib.request.urlopen(RELEASE) as r:
        zipfile.ZipFile(io.BytesIO(r.read())).extractall(out)
    return out


def read(ocr):
    return next(iter(ocr.predict(str(ROOT / "demo/examples/receipt_1.jpg"))))["rec_texts"]


def test_stock_and_finetuned_read_a_receipt(models):
    stock = read(paddleocr.PaddleOCR(**COMMON))
    tuned = read(paddleocr.PaddleOCR(**COMMON, text_detection_model_dir=str(models / "det"),
                                     text_recognition_model_dir=str(models / "rec")))
    assert len(stock) > 5 and len(tuned) > 5
    assert any("28,000" in t for t in tuned)
