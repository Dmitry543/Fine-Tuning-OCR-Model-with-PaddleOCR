# Fine-tune PaddleOCR PP-OCRv6 on your own data

[English](README.md) | [简体中文](README.zh-CN.md)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Dmitry543/Fine-Tuning-OCR-Model-with-PaddleOCR/blob/main/finetune.ipynb)
![PaddleOCR 3.7](https://img.shields.io/badge/PaddleOCR-3.7-blue)
![PP-OCRv6](https://img.shields.io/badge/model-PP--OCRv6-blue)
![License](https://img.shields.io/badge/license-Apache--2.0-green)
[![tests](https://github.com/Dmitry543/Fine-Tuning-OCR-Model-with-PaddleOCR/actions/workflows/tests.yml/badge.svg)](https://github.com/Dmitry543/Fine-Tuning-OCR-Model-with-PaddleOCR/actions/workflows/tests.yml)

Fine-tune the PP-OCRv6 text detector and recognizer on your data, then measure whether it actually got better. One command does the whole loop: prepare data, measure the stock models, train, export, measure again.

## Results

PP-OCRv6_small fine-tuned on [CORD](https://huggingface.co/datasets/naver-clova-ix/cord-v2) receipts (800 training images), measured on the 100 held-out test receipts:

| Metric | Stock | Fine-tuned | Change |
|---|---|---|---|
| **End-to-end F1** (box and text both right) | 79.1% | **88.2%** | +9.1 pt |
| Detection F1 | 89.7% | **95.8%** | +6.1 pt |
| Detection precision / recall | 87.0% / 92.6% | **93.9% / 97.7%** | +6.9 / +5.1 pt |
| Recognition accuracy (1,555 lines) | 90.2% | **94.5%** | +4.3 pt |
| Character error rate | 1.79% | **1.10%** | −38% |
| Speed (T4, per receipt) | 0.14 s | 0.14 s | same model size |

<p align="center"><img src="docs/assets/comparison.jpg" width="100%" alt="Stock vs fine-tuned PP-OCRv6 on CORD test receipts"></p>
<p align="center"><sub>The two test receipts with the largest gain. Left: stock model. Right: fine-tuned. Green: box and text correct. Red: wrong, with the predicted text below. The stock detector breaks dot-matrix print into single dots.</sub></p>

Numbers are from a single training run. Training took 54 minutes on a free Colab T4 (detection 20 epochs, 30 min; recognition 10 epochs, 24 min); the whole notebook, including setup and evaluation, about 1.5 hours. Recognition accuracy on the validation split was still rising at the last epoch, so longer training may help on your data. The same models evaluated on an Apple M2 CPU score within 0.1 pt (1.6 s per receipt). Raw numbers: [`results/cord/`](results/cord).

### PaddleOCR-VL-1.6 with LoRA

The same 1,555 test lines read by PaddleOCR-VL-1.6 (0.9B vision-language model), before and after a 21-minute LoRA fine-tune on a T4 ([guide](docs/paddleocr-vl.md), [notebook](finetune_vl.ipynb)):

| Line recognition | Accuracy | Character error rate | Time per line (T4) |
|---|---|---|---|
| PaddleOCR-VL-1.6, stock | 83.4% | 12.9% | 932 ms |
| PaddleOCR-VL-1.6 + LoRA | 90.8% | 4.2% | 874 ms |
| PP-OCRv6_small, stock | 90.2% | 1.8% | 9 ms |
| PP-OCRv6_small, fine-tuned | **94.5%** | **1.1%** | **8 ms** |

LoRA cuts the VL model's errors by two thirds, but on receipt lines the fine-tuned PP-OCRv6 recognizer is still more accurate and about 100× faster.

## Quick start

**Colab (free GPU):** open the notebook above and run all cells.

**Your own machine** (Linux with an NVIDIA GPU, or any machine for a CPU smoke test):

```bash
git clone https://github.com/Dmitry543/Fine-Tuning-OCR-Model-with-PaddleOCR.git
cd Fine-Tuning-OCR-Model-with-PaddleOCR
python3 -m venv .venv && source .venv/bin/activate

bash setup.sh --gpu     # PaddlePaddle 3.3.1, PaddleOCR 3.7.0, PP-OCRv6 pretrained weights
bash finetune.sh        # CORD: download, baseline, train, export, compare
```

The comparison table is written to `results/cord/comparison.md`, the models to `output/inference/`.

## Train on your own data

Label images with [PPOCRLabel](https://github.com/PFCCLab/PPOCRLabel), then:

```bash
python tools/prepare_data.py ppocrlabel --input path/to/labelled_folder --out data/mydata
DATA=data/mydata bash finetune.sh
```

Labelling conventions, the label format, the character dictionary and other datasets: [docs/custom-data.md](docs/custom-data.md).

## Use the fine-tuned models

```python
from paddleocr import PaddleOCR

ocr = PaddleOCR(
    text_detection_model_name="PP-OCRv6_small_det", text_detection_model_dir="output/inference/det",
    text_recognition_model_name="PP-OCRv6_small_rec", text_recognition_model_dir="output/inference/rec",
    use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False,
)
for res in ocr.predict("receipt.jpg"):
    print(res["rec_texts"])
```

**ONNX** (ONNX Runtime instead of PaddlePaddle at inference time):

```bash
paddlex --install paddle2onnx     # Python 3.12 or older; there are no 3.13 builds yet
python tools/export_onnx.py output/inference/det output/inference/rec   # -> output/onnx/{det,rec}
python tools/evaluate.py --name onnx --det output/onnx/det --rec output/onnx/rec --engine onnxruntime
```

Load it with the snippet above, pointing at `output/onnx/...` and adding `engine="onnxruntime"`. On the CORD test set the ONNX models score exactly the same as the Paddle ones and run 4.5× faster on an Apple M2 CPU (0.36 s vs 1.63 s per receipt). Two PaddlePaddle 3.3 details otherwise break the conversion; `finetune.sh` and `tools/export_onnx.py` handle both (see [troubleshooting](docs/troubleshooting.md)).

**Demo:** `pip install gradio && python demo/app.py` opens a page that runs the stock and fine-tuned models side by side on any image you upload.

## How it is measured

`tools/evaluate.py` runs the same PaddleOCR 3.x inference code you deploy with and reports:

| Metric | Meaning |
|---|---|
| Detection F1 | Predicted text-line boxes matched one-to-one to labelled boxes at IoU ≥ 0.5 |
| Recognition accuracy | Exact text on the labelled line crops (detection errors excluded) |
| Character error rate | Edit distance / label length on the same crops |
| End-to-end F1 | A prediction counts only if its box matches **and** its text is exact |

Text comparison ignores whitespace (CORD writes `( L )` where the model reads `(L)`); case and punctuation count. Regions labelled as unreadable are ignored. Checkpoints are selected on the validation split; the test split is only used for the numbers above.

## What is in the repo

```
setup.sh                 install pinned PaddlePaddle / PaddleOCR, download pretrained weights
finetune.sh              baseline → train det → train rec → export → compare
finetune.ipynb           the same steps, explained, for Colab
finetune_vl.ipynb        LoRA fine-tuning of PaddleOCR-VL-1.6 (Colab)
configs/                 PP-OCRv6_small det/rec fine-tuning configs
tools/prepare_data.py    CORD, PPOCRLabel and Total-Text → PaddleOCR labels
tools/evaluate.py        metrics and stock-vs-fine-tuned table
tools/visualize.py       side-by-side comparison images
tools/export_onnx.py     ONNX export
demo/app.py              Gradio demo
vl/                      PaddleOCR-VL LoRA training and evaluation
docs/                    custom data, PaddleOCR-VL, troubleshooting
tests/                   converter and metric checks (run in CI)
```

## Notes

- Training needs an NVIDIA GPU. PaddlePaddle has no Apple-silicon GPU support, so on a Mac use Colab for training; data preparation, evaluation, ONNX export and the demo run fine on the CPU.
- Versions are pinned (PaddlePaddle 3.3.1, PaddleOCR v3.7.0) because PaddleOCR's training configs and inference API change between releases.
- Problems: [docs/troubleshooting.md](docs/troubleshooting.md).

## Data and license

Code: [Apache-2.0](LICENSE). The CORD dataset is © NAVER Clova, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); the receipts in `demo/examples/` and `docs/assets/` come from it.

```bibtex
@inproceedings{park2019cord,
  title     = {CORD: A Consolidated Receipt Dataset for Post-OCR Parsing},
  author    = {Park, Seunghyun and Shin, Seung and Lee, Bado and Lee, Junyeop and Surh, Jaeheung and Seo, Minjoon and Lee, Hwalsuk},
  booktitle = {Document Intelligence Workshop at NeurIPS},
  year      = {2019}
}
```
