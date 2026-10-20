"""Convert exported PaddleOCR models to ONNX.

    paddlex --install paddle2onnx    # once
    python tools/export_onnx.py output/inference/det output/inference/rec --out output/onnx

Each output folder holds inference.onnx and inference.yml and loads with engine="onnxruntime":

    PaddleOCR(text_detection_model_name="PP-OCRv6_small_det", text_detection_model_dir="output/onnx/det",
              text_recognition_model_name="PP-OCRv6_small_rec", text_recognition_model_dir="output/onnx/rec",
              engine="onnxruntime")

PaddlePaddle 3.3 writes some float attributes as 64-bit, which Paddle2ONNX 2.0 rejects
("the 0th elementwise MUST be ir::FloatAttribute"). They are small constants (1.0, 2.0, ...),
so a copy of the model is rewritten with 32-bit attributes before conversion.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def convert(model_dir: Path, out_dir: Path, opset: int):
    with tempfile.TemporaryDirectory() as tmp:
        patched = Path(tmp) / model_dir.name
        shutil.copytree(model_dir, patched)
        graph = patched / "inference.json"
        graph.write_text(graph.read_text().replace('"0.a_f64","D":', '"0.a_f32","D":'))
        out_dir.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, PATH=f"{Path(sys.executable).parent}{os.pathsep}{os.environ['PATH']}")
        subprocess.run(["paddlex", "--paddle2onnx", "--paddle_model_dir", str(patched),
                        "--onnx_model_dir", str(out_dir), "--opset_version", str(opset)], check=True, env=env)
    if not (out_dir / "inference.onnx").exists():
        raise SystemExit(f"conversion failed for {model_dir}")
    print(f"{model_dir} -> {out_dir / 'inference.onnx'}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model_dirs", nargs="+", help="folders produced by tools/export_model.py")
    p.add_argument("--out", default="output/onnx")
    p.add_argument("--opset", type=int, default=11)
    args = p.parse_args()
    for d in map(Path, args.model_dirs):
        convert(d, Path(args.out) / d.name, args.opset)


if __name__ == "__main__":
    main()
