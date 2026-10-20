#!/usr/bin/env bash
# Install PaddlePaddle, PaddleOCR (pinned) and the pretrained PP-OCRv6 weights.
#
#   bash setup.sh          # CPU (macOS, Linux, Windows WSL)
#   bash setup.sh --gpu    # NVIDIA GPU (Colab, Kaggle, cloud); picks the CUDA 13 or 12.6 build from the driver
set -euo pipefail

PADDLE_VERSION=3.3.1
PADDLEOCR_REF=v3.7.0
WEIGHTS=https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model
PY=${PYTHON:-python3}

if [[ "${1:-}" == "--gpu" ]]; then
  # Match the driver: a CUDA 12 wheel on a CUDA 13 machine (e.g. Colab) installs an older NCCL that
  # breaks the preinstalled PyTorch, which PaddleX imports through modelscope.
  driver_cuda=$(nvidia-smi | grep -o 'CUDA Version: [0-9]*' | grep -o '[0-9]*$')
  cuda=cu126; [[ "${driver_cuda:-12}" -ge 13 ]] && cuda=cu130
  $PY -m pip install -q "paddlepaddle-gpu==$PADDLE_VERSION" -i "https://www.paddlepaddle.org.cn/packages/stable/$cuda/"
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
