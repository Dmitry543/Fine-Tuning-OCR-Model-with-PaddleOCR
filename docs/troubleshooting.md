# Troubleshooting

**`zsh: no matches found: Train.dataset.label_file_list=[...]`**
zsh treats `[...]` as a file pattern. Quote the override: `'Train.dataset.label_file_list=[data/mydata/det_train.txt]'`. `finetune.sh` runs in bash and is not affected.

**`No Images in train dataset, please ensure ...`**
The training list has fewer samples than one batch, or the paths in it do not resolve against `data_dir`. Lower `Train.loader.batch_size_per_card`, or check that `Train.dataset.data_dir` is the folder that contains `images/` and `rec/`.

**`output/det/best_accuracy` does not exist after training**
PaddleOCR saves `best_accuracy` only when it evaluates, every `Global.eval_batch_step` iterations. With a small dataset, training can finish before the first evaluation. Set `'Global.eval_batch_step=[0,N]'` with N below the total number of iterations (iterations per epoch × epochs; PaddleOCR logs the iterations per epoch at startup).

**`Out of memory` on GPU**
Lower `Train.loader.batch_size_per_card` (detection 8 → 4, recognition 64 → 32). For recognition, also lower `Train.sampler.first_bs` to the same value.

**`ConvertPirAttribute2RuntimeAttribute not support [pir::ArrayAttribute<pir::DoubleAttribute>]` on a Linux or Windows CPU**
PaddlePaddle 3.3.1's oneDNN (MKL-DNN) executor cannot run PP-OCRv6, including the stock models. Pass `enable_mkldnn=False` to `PaddleOCR(...)`; the tools and demo in this repo already do.

**Training is very slow on a Mac**
PaddlePaddle has no Apple GPU support, so training runs on the CPU at a few minutes per recognition batch. Use the Colab notebook for real training; the Mac is fine for data preparation, evaluation and the demo.

**`Checking connectivity to the model hosters` hangs**
Set `PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True`. If Hugging Face is blocked on your network, also set `PADDLE_PDX_MODEL_SOURCE=BOS` to download official models from Baidu's servers.

**ONNX conversion fails or hangs**
Two PaddlePaddle 3.3 export details break Paddle2ONNX 2.0:
- 64-bit float attributes (`the 0th elementwise MUST be ir::FloatAttribute`). `tools/export_onnx.py` rewrites them.
- On CUDA builds, linear layers are exported as `linear_v2`, which Paddle2ONNX does not support. It aborts with `Due to the unsupported operators, the conversion is aborted`, and on macOS the crash handler can then hang instead of exiting. Export with `FLAGS_use_legacy_linear=1` (as `finetune.sh` does) and convert again.

Paddle2ONNX has no Python 3.13 builds yet, so convert in a Python 3.12 (or older) environment.
