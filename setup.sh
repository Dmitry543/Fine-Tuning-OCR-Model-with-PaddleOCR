#!/usr/bin/env bash
# Install PaddlePaddle, PaddleOCR (pinned) and the pretrained PP-OCRv6 weights.
#
#   bash setup.sh          # CPU (macOS, Linux, Windows WSL)
#   bash setup.sh --gpu    # NVIDIA GPU, CUDA 12.x driver (Colab, Kaggle, most cloud GPUs)
set -euo pipefail

PADDLE_VERSION=3.3.1
PADDLEOCR_REF=v3.7.0
WEIGHTS=https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model
PY=${PYTHON:-python3}

if [[ "${1:-}" == "--gpu" ]]; then
  $PY -m pip install -q "paddlepaddle-gpu==$PADDLE_VERSION" -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
else
  $PY -m pip install -q "paddlepaddle==$PADDLE_VERSION"
fi

# Training code is not in the pip package, so it comes from the matching PaddleOCR release.
if [[ ! -d third_party/PaddleOCR ]]; then
  git clone -q --depth 1 --branch "$PADDLEOCR_REF" https://github.com/PaddlePaddle/PaddleOCR.git third_party/PaddleOCR
fi
$PY -m pip install -q -r third_party/PaddleOCR/requirements.txt -r requirements.txt

mkdir -p pretrained
for task in det rec; do
  name="PP-OCRv6_small_${task}_pretrained.pdparams"
  [[ -f "pretrained/$name" ]] || curl -fsSL -o "pretrained/$name" "$WEIGHTS/$name"
done

$PY -c "import paddle; paddle.utils.run_check()"
