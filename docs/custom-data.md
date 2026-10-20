# Train on your own data

## 1. Label

Use [PPOCRLabel](https://github.com/PFCCLab/PPOCRLabel). It pre-labels images with PP-OCR, so you mostly fix boxes and text instead of drawing them.

- Draw one box per **text line**, the same unit PP-OCR detects. Split a line where there is a large gap (for example, between an item name and its price).
- Tick **difficult** for text you cannot read. It becomes an ignored region: not penalised in detection, not used for recognition.
- Export with **File → Export Label**. The folder now contains `Label.txt`.

## 2. Convert

```bash
python tools/prepare_data.py ppocrlabel --input path/to/labelled_folder --out data/mydata
```

Images are split 80/10/10 into train, validation and test (`--val-ratio`, `--test-ratio`, `--seed`). The output layout is the same for every source:

```
data/mydata/
  images/{train,val,test}/   full images (longest side capped at 1600 px)
  det_{split}.txt            image<TAB>[{"transcription": "...", "points": [[x, y], ...]}, ...]
  rec/{split}/               rectified line crops
  rec_{split}.txt            crop<TAB>text
```

## 3. Train and compare

```bash
DATA=data/mydata bash finetune.sh
```

Results are written to `results/mydata/`.

## Characters and line length

PP-OCRv6 recognises the characters in `third_party/PaddleOCR/ppocr/utils/dict/ppocrv6_dict.txt` (about 18,700: Chinese, Japanese, English and 46 Latin-script languages). During training, PaddleOCR silently deletes characters that are not in the dictionary from your labels, and skips lines longer than `max_text_length` (25). Check your labels against the dictionary before training:

```bash
python - <<'PY'
chars = set(open("third_party/PaddleOCR/ppocr/utils/dict/ppocrv6_dict.txt", encoding="utf-8").read().split("\n"))
text = "".join(l.split("\t", 1)[1] for l in open("data/mydata/rec_train.txt", encoding="utf-8"))
print(sorted(set(text) - chars - {" ", "\n"}))
PY
```

To add characters, append them to a copy of the dictionary and point `Global.character_dict_path` at it. The recognizer's output layer changes size, so those weights are trained from scratch: expect to need more data and epochs.

## Other sources

**Total-Text** (curved scene text) is supported, but its [terms](https://github.com/cs-chan/Total-Text-Dataset/blob/master/DATASET_TERMS.md) allow non-commercial academic use only and forbid redistribution, so download it yourself and keep the original layout (`Train/`, `Test/`, `Annotation/groundtruth_polygonal_annotation/`):

```bash
python tools/prepare_data.py totaltext --input path/to/Total-Text --out data/totaltext
```

Total-Text has no validation split; `finetune.sh` then selects checkpoints on the test split, so treat its numbers as optimistic.

Any other dataset: write the two label files in the format above and point `DATA` at the folder.
