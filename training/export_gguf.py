# /// script
# requires-python = ">=3.11"
# dependencies = ["torch", "transformers", "peft", "accelerate", "huggingface_hub", "sentencepiece", "protobuf", "numpy", "pillow", "torchvision"]
# ///
"""Merge the student LoRA into the base model and export base and student as GGUF.

Both models go through the same conversion and quantization, so the untuned and fine-tuned
students are served identically during evaluation. Runs on a CPU HF Job:

    hf jobs uv run --flavor cpu-upgrade --secrets HF_TOKEN \
      -e BASE=google/gemma-4-E4B-it -e ADAPTER=user/lora -e OUT_REPO=user/gguf training/export_gguf.py
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    base_id = os.environ.get("BASE", "google/gemma-4-E4B-it")
    adapter_id = os.environ["ADAPTER"]
    out_repo = os.environ["OUT_REPO"]
    outtype = os.environ.get("OUTTYPE", "q8_0")
    work = Path("/tmp/export")
    work.mkdir(parents=True, exist_ok=True)

    import torch
    from huggingface_hub import HfApi, snapshot_download
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

    base_dir = Path(snapshot_download(base_id))
    try:
        model = AutoModelForImageTextToText.from_pretrained(base_dir, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    except (ValueError, KeyError):  # text-only checkpoints
        model = AutoModelForCausalLM.from_pretrained(base_dir, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    model = PeftModel.from_pretrained(model, adapter_id)
    model = model.merge_and_unload()
    merged_dir = work / "merged"
    model.save_pretrained(merged_dir, safe_serialization=True)
    try:
        AutoProcessor.from_pretrained(base_dir).save_pretrained(merged_dir)
    except Exception:
        AutoTokenizer.from_pretrained(base_dir).save_pretrained(merged_dir)
    for name in ("chat_template.jinja", "tokenizer.model", "tokenizer.json", "tokenizer_config.json"):
        if (base_dir / name).exists() and not (merged_dir / name).exists():
            shutil.copy(base_dir / name, merged_dir / name)
    del model

    run(["git", "clone", "--depth", "1", "https://github.com/ggml-org/llama.cpp", str(work / "llama.cpp")])
    # the converter's only extra dependency is llama.cpp's own gguf package, which ships in the repo
    os.environ["PYTHONPATH"] = str(work / "llama.cpp" / "gguf-py") + os.pathsep + os.environ.get("PYTHONPATH", "")
    convert = str(work / "llama.cpp" / "convert_hf_to_gguf.py")
    stem = os.environ.get("NAME", base_id.split("/")[-1])
    outputs = {
        f"{stem}-base-{outtype}.gguf": base_dir,
        f"{stem}-apprentice-{outtype}.gguf": merged_dir,
    }
    api = HfApi()
    for name, src in outputs.items():
        out = work / name
        run([sys.executable, convert, str(src), "--outtype", outtype, "--outfile", str(out)])
        api.upload_file(repo_id=out_repo, path_or_fileobj=str(out), path_in_repo=name, commit_message=f"add {name}")
        print(f"uploaded {name}", flush=True)
        out.unlink()


if __name__ == "__main__":
    main()
