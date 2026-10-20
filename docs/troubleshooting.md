# Troubleshooting

**`zsh: no matches found: Train.dataset.label_file_list=[...]`**
zsh treats `[...]` as a file pattern. Quote the override: `'Train.dataset.label_file_list=[data/mydata/det_train.txt]'`. `finetune.sh` runs in bash and is not affected.

**`No Images in train dataset, please ensure ...`**
The training list has fewer samples than one batch, or the paths in it do not resolve against `data_dir`. Lower `Train.loader.batch_size_per_card`, or check that `Train.dataset.data_dir` is the folder that contains `images/` and `rec/`.

**`output/det/best_accuracy` does not exist after training**
PaddleOCR saves `best_accuracy` only when it evaluates, every `Global.eval_batch_step` iterations. With a small dataset, training can finish before the first evaluation. Set `'Global.eval_batch_step=[0,N]'` with N below the total number of iterations (iterations per epoch × epochs; PaddleOCR logs the iterations per epoch at startup).

**`Out of memory` on GPU**
Lower `Train.loader.batch_size_per_card` (detection 8 → 4, recognition 64 → 32). For recognition, also lower `Train.sampler.first_bs` to the same value.

**Training is very slow on a Mac**
PaddlePaddle has no Apple GPU support, so training runs on the CPU at a few minutes per recognition batch. Use the Colab notebook for real training; the Mac is fine for data preparation, evaluation and the demo.

**`Checking connectivity to the model hosters` hangs**
Set `PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True`. If Hugging Face is blocked on your network, also set `PADDLE_PDX_MODEL_SOURCE=BOS` to download official models from Baidu's servers.
