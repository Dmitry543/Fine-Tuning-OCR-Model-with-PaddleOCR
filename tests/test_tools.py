"""Fast checks for the data converter and the metrics. No PaddlePaddle or GPU needed."""

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from evaluate import match, norm, prf  # noqa: E402
from prepare_data import IGNORE, cord_lines, crop_text  # noqa: E402


def word(text, x, y, w=40, h=20, row=1):
    return {"text": text, "row_id": row,
            "quad": {"x1": x, "y1": y, "x2": x + w, "y2": y, "x3": x + w, "y3": y + h, "x4": x, "y4": y + h}}


def test_cord_lines_merge_close_words_and_split_columns():
    gt = {"valid_line": [{"words": [word("Ice", 10, 10), word("Tea", 55, 10), word("24,000", 300, 10)]}],
          "repeating_symbol": [[{"quad": word("-", 0, 40, w=400)["quad"], "text": "-"}]], "dontcare": []}
    regions = cord_lines(gt, gap=1.0)
    texts = sorted(t for t, _ in regions)
    assert texts == [IGNORE, "24,000", "Ice Tea"]


def test_crop_text_rectifies_rotated_line():
    image = np.full((200, 300, 3), 255, np.uint8)
    poly = np.array([[50, 100], [250, 60], [255, 85], [55, 125]], float)
    crop = crop_text(image, poly)
    assert crop is not None and crop.shape[1] > crop.shape[0] * 4


def test_match_counts_ignore_regions_and_iou():
    gts = [{"transcription": "TOTAL", "points": [[0, 0], [100, 0], [100, 20], [0, 20]]},
           {"transcription": IGNORE, "points": [[0, 50], [100, 50], [100, 70], [0, 70]]}]
    preds = [{"points": [[2, 1], [98, 1], [98, 19], [2, 19]], "text": "TOTAL"},
             {"points": [[0, 50], [100, 50], [100, 70], [0, 70]], "text": "noise"}]
    n_pred, n_gt, matched = match(preds, gts)
    assert (n_pred, n_gt, matched) == (1, 1, [("TOTAL", "TOTAL")])
    assert prf(1, n_pred, n_gt)["f1"] == 1.0


def test_norm_ignores_whitespace_only():
    assert norm("( L )") == norm("(L)")
    assert norm("Total") != norm("TOTAL")


def test_prepare_ppocrlabel_end_to_end(tmp_path):
    folder = tmp_path / "labelled"
    folder.mkdir()
    lines = []
    for i in range(10):
        image = np.full((120, 300, 3), 255, np.uint8)
        cv2.putText(image, f"LINE {i}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
        cv2.imwrite(str(folder / f"img{i}.jpg"), image)
        ann = [{"transcription": f"LINE {i}", "points": [[15, 25], [200, 25], [200, 75], [15, 75]], "difficult": False}]
        lines.append(f"labelled/img{i}.jpg\t{json.dumps(ann)}")
    (folder / "Label.txt").write_text("\n".join(lines))

    out = tmp_path / "data"
    subprocess.run([sys.executable, str(ROOT / "tools/prepare_data.py"), "ppocrlabel",
                    "--input", str(folder), "--out", str(out)], check=True, capture_output=True)

    counts = {f.name: len(f.read_text().splitlines()) for f in out.glob("*.txt")}
    assert counts == {"det_train.txt": 8, "det_val.txt": 1, "det_test.txt": 1,
                      "rec_train.txt": 8, "rec_val.txt": 1, "rec_test.txt": 1}
    rel, text = (out / "rec_train.txt").read_text().splitlines()[0].split("\t")
    assert (out / rel).exists() and text.startswith("LINE ")


def test_prepare_refuses_non_empty_output(tmp_path):
    (tmp_path / "x").write_text("")
    result = subprocess.run([sys.executable, str(ROOT / "tools/prepare_data.py"), "cord", "--out", str(tmp_path)],
                            capture_output=True, text=True)
    assert result.returncode != 0 and "not empty" in result.stderr


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
