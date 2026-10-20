#!/usr/bin/env bash
# Fine-tune PP-OCRv6 detection and recognition, export both, and compare against the stock models.
#
#   bash finetune.sh                       # GPU, CORD receipts
#   DEVICE=cpu bash finetune.sh            # CPU (slow: use for a smoke test only)
#   DATA=data/mydata bash finetune.sh      # your own data, prepared with tools/prepare_data.py
#
# Any extra arguments are passed to both training runs as PaddleOCR overrides, e.g.
#   bash finetune.sh Global.epoch_num=10
set -euo pipefail
set -f  # overrides contain [..] lists; never glob them

DEVICE=${DEVICE:-gpu}
DATA=${DATA:-data/cord}
PY=${PYTHON:-python3}
OVERRIDES=("$@")
RESULTS=results/$(basename "$DATA")
export PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True

USE_GPU=True; EVAL_DEVICE=gpu:0
[[ "$DEVICE" == cpu ]] && { USE_GPU=False; EVAL_DEVICE=cpu; }

[[ -f "$DATA/det_train.txt" ]] || $PY tools/prepare_data.py cord --out "$DATA"

data_args() {  # point a config at $DATA; validation split if it exists, else test
  local task=$1 eval_split=val
  [[ -f "$DATA/${task}_val.txt" ]] || eval_split=test
  echo "Train.dataset.data_dir=$DATA/" "Eval.dataset.data_dir=$DATA/" \
       "Train.dataset.label_file_list=[$DATA/${task}_train.txt]" \
       "Eval.dataset.label_file_list=[$DATA/${task}_${eval_split}.txt]"
}

echo "== baseline: stock PP-OCRv6_small"
[[ -f "$RESULTS/stock.json" ]] || $PY tools/evaluate.py --name stock --data "$DATA" --device "$EVAL_DEVICE" --out-dir "$RESULTS" \
  --det PP-OCRv6_small_det --rec PP-OCRv6_small_rec

for task in det rec; do
  echo "== training $task"
  # shellcheck disable=SC2046
  $PY third_party/PaddleOCR/tools/train.py -c "configs/${task}_ppocrv6_small.yml" \
    -o Global.use_gpu=$USE_GPU $(data_args $task) ${OVERRIDES[@]+"${OVERRIDES[@]}"}
  $PY third_party/PaddleOCR/tools/export_model.py -c "configs/${task}_ppocrv6_small.yml" \
    -o Global.use_gpu=$USE_GPU Global.pretrained_model="output/$task/best_accuracy" \
       Global.save_inference_dir="output/inference/$task"
done

echo "== fine-tuned"
$PY tools/evaluate.py --name finetuned --data "$DATA" --device "$EVAL_DEVICE" --out-dir "$RESULTS" \
  --det output/inference/det --rec output/inference/rec
$PY tools/evaluate.py --compare "$RESULTS/stock.json" "$RESULTS/finetuned.json" | tee "$RESULTS/comparison.md"
