# LoRA fine-tuning for PaddleOCR-VL-1.6

[PaddleOCR-VL-1.6](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6) is a 0.9B vision-language model (NaViT-style vision encoder + ERNIE-4.5-0.3B). It reads a cropped element (a text line, table, formula, chart or seal) and writes out its content. This track fine-tunes it for text-line recognition with LoRA and compares it with PP-OCRv6 on the same line crops.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Dmitry543/Fine-Tuning-OCR-Model-with-PaddleOCR/blob/main/finetune_vl.ipynb)

## Why LoRA, and why Transformers

PaddlePaddle's official recipe (ERNIEKit SFT) does full fine-tuning in bfloat16 with FlashAttention and is documented on an 80 GB A800. A free T4 has 16 GB and supports neither bfloat16 nor FlashAttention 2. Transformers 5 loads PaddleOCR-VL natively, so this track trains LoRA adapters (0.7% of the weights) with PEFT: weights in float32, forward pass in float16 autocast.

## Run

```bash
pip install -r vl/requirements.txt
python tools/prepare_data.py cord --out data/cord     # skip if already prepared

python vl/evaluate.py --name vl_stock                   # stock model on the test crops
python vl/train_lora.py --out output/vl_lora            # 1 epoch, ~13,000 lines
python vl/evaluate.py --name vl_lora --adapter output/vl_lora
```

Options: `--epochs`, `--lr` (2e-4), `--rank` (16), `--batch-size` (8), `--limit` for a quick smoke test. Your own data works the same way: any folder written by `tools/prepare_data.py` (`--data data/mydata`).

## Use the adapter

```python
import sys, torch
sys.path.insert(0, "vl")
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor
from data import MODEL, generation_batch

processor = AutoProcessor.from_pretrained(MODEL)
model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.float16).to("cuda").eval()
model = PeftModel.from_pretrained(model, "output/vl_lora").merge_and_unload()

batch = generation_batch(processor, ["line.jpg"]).to("cuda")
out = model.generate(**batch, max_new_tokens=64)
print(processor.decode(out[0, batch["input_ids"].shape[1]:], skip_special_tokens=True))
```

## Results and when to use which

On the 1,555 CORD test lines (T4):

| | Accuracy | Character error rate | Time per line |
|---|---|---|---|
| PaddleOCR-VL-1.6, stock | 83.4% | 12.9% | 932 ms |
| PaddleOCR-VL-1.6 + LoRA (1 epoch, 21 min) | 90.8% | 4.2% | 874 ms |
| PP-OCRv6_small, fine-tuned | 94.5% | 1.1% | 8 ms |

Errors seen in a sample of the stock model's output: decimal separators (`60.000` read as `60,000`), faded characters (`AVOCADO COFFEE` → `AVOGADO CORFEE`), and a lone digit answered as LaTeX (`1` → `\(x_{1}=2\)`). LoRA cuts the character error rate by two thirds. For plain text lines like these, fine-tuned PP-OCRv6 is the better choice. PaddleOCR-VL is worth fine-tuning when you need what PP-OCRv6 cannot do, such as tables, formulas or charts from the same model, and the scripts here measure whether it pays off on your data.
