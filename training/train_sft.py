"""LoRA fine-tune of Gemma 4 E4B on Beam's test-verified agent sessions. Built for a free Colab T4.

Each training example is one whole agent session: the task prompt, then a single model turn
that interleaves Beam's thinking, tool calls and the tool results. The loss covers what the
model writes (thinking and tool calls) and skips the tool results, so the student learns to
act and reason, not to invent command output.

Configured by environment variables so it runs unchanged under `colab exec -f`:
    SFT_DATA=/content/sft.jsonl  SFT_OUT=/content/out  SFT_MODEL=unsloth/gemma-4-E4B-it
    SFT_MAX_LEN=8192  SFT_EPOCHS=2  SFT_LR=1e-4  SFT_RANK=16  SFT_MAX_STEPS=0  SFT_THINKING=1
"""

from __future__ import annotations

import json
import os
import re

MODEL_TURN_OPEN = "<|turn>model\n"
TURN_CLOSE = "<turn|>"
TOOL_RESPONSE_OPEN = "<|tool_response>"
TOOL_RESPONSE_CLOSE = "<tool_response|>"


def train_char_spans(text: str) -> list[tuple[int, int]]:
    """Character ranges the loss should cover: model turns, minus tool-response blocks."""
    spans = []
    for m in re.finditer(re.escape(MODEL_TURN_OPEN), text):
        start = m.end()
        end = text.find(TURN_CLOSE, start)
        end = len(text) if end == -1 else end + len(TURN_CLOSE)
        cursor = start
        while True:
            open_at = text.find(TOOL_RESPONSE_OPEN, cursor, end)
            if open_at == -1:
                spans.append((cursor, end))
                break
            spans.append((cursor, open_at))
            close_at = text.find(TOOL_RESPONSE_CLOSE, open_at, end)
            if close_at == -1:
                break
            cursor = close_at + len(TOOL_RESPONSE_CLOSE)
    return [(a, b) for a, b in spans if b > a]


def labels_from_offsets(input_ids: list[int], offsets: list[tuple[int, int]], spans) -> list[int]:
    labels = []
    j = 0
    for tok, (a, b) in zip(input_ids, offsets):
        while j < len(spans) and spans[j][1] <= a:
            j += 1
        inside = j < len(spans) and spans[j][0] <= a < spans[j][1] and b > a
        labels.append(tok if inside else -100)
    return labels


def prepare_messages(example: dict, thinking: bool) -> list[dict]:
    """Template expects tool-call arguments as objects; optionally drop the teacher's thinking."""
    out = []
    for m in example["messages"]:
        m = dict(m)
        if m.get("tool_calls"):
            calls = []
            for tc in m["tool_calls"]:
                tc = json.loads(json.dumps(tc))
                args = tc["function"].get("arguments") or "{}"
                tc["function"]["arguments"] = json.loads(args) if isinstance(args, str) else args
                calls.append(tc)
            m["tool_calls"] = calls
        if not thinking:
            m.pop("reasoning_content", None)
        out.append(m)
    return out


def _start_memory_log(every_s: int = 20) -> None:
    """Print host RAM and GPU memory periodically, so an out-of-memory death leaves a trail."""
    import subprocess
    import threading
    import time

    def loop():
        while True:
            try:
                ram = open("/proc/meminfo").read().split("\n")
                info = {l.split(":")[0]: int(l.split()[1]) // 1024 for l in ram if l.startswith(("MemTotal", "MemAvailable"))}
                gpu = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader"],
                                     capture_output=True, text=True).stdout.strip()
                print(f"[mem] ram_available={info.get('MemAvailable')}MB/{info.get('MemTotal')}MB gpu_used={gpu}",
                      flush=True)
            except Exception:
                pass
            time.sleep(every_s)

    threading.Thread(target=loop, daemon=True).start()


def main() -> None:
    data_path = os.environ.get("SFT_DATA", "/content/sft.jsonl")
    out_dir = os.environ.get("SFT_OUT", "/content/out")
    model_id = os.environ.get("SFT_MODEL", "unsloth/gemma-4-E4B-it")
    max_len = int(os.environ.get("SFT_MAX_LEN", "8192"))
    epochs = float(os.environ.get("SFT_EPOCHS", "2"))
    lr = float(os.environ.get("SFT_LR", "1e-4"))
    rank = int(os.environ.get("SFT_RANK", "16"))
    max_steps = int(os.environ.get("SFT_MAX_STEPS", "0"))
    thinking = os.environ.get("SFT_THINKING", "1") == "1"

    _start_memory_log()

    import torch
    from unsloth import FastModel  # import before transformers so its patches apply

    from datasets import Dataset
    from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments

    model, tokenizer = FastModel.from_pretrained(
        model_id, max_seq_length=max_len, load_in_4bit=True, full_finetuning=False
    )
    tok = getattr(tokenizer, "tokenizer", tokenizer)  # multimodal processors wrap the text tokenizer

    rows, dropped, lengths = [], 0, []
    for line in open(data_path):
        ex = json.loads(line)
        text = tok.apply_chat_template(
            prepare_messages(ex, thinking), tools=ex["tools"], tokenize=False,
            enable_thinking=thinking, preserve_thinking=thinking,
        )
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        n = len(enc["input_ids"])
        lengths.append(n)
        if n > max_len:
            dropped += 1
            continue
        labels = labels_from_offsets(enc["input_ids"], enc["offset_mapping"], train_char_spans(text))
        if sum(l != -100 for l in labels) == 0:
            dropped += 1
            continue
        rows.append({"input_ids": enc["input_ids"], "attention_mask": [1] * n, "labels": labels})
    lengths.sort()
    print(json.dumps({
        "examples": len(rows), "dropped_too_long_or_empty": dropped,
        "tokens_p50": lengths[len(lengths) // 2] if lengths else 0, "tokens_max": lengths[-1] if lengths else 0,
        "trainable_tokens": sum(sum(l != -100 for l in r["labels"]) for r in rows),
    }), flush=True)
    if not rows:
        raise SystemExit("no trainable examples; raise SFT_MAX_LEN or check the data")

    model = FastModel.get_peft_model(
        model,
        finetune_vision_layers=False, finetune_language_layers=True,
        finetune_attention_modules=True, finetune_mlp_modules=True,
        r=rank, lora_alpha=rank * 2, lora_dropout=0.0, bias="none",
        use_gradient_checkpointing="unsloth", random_state=0,
    )

    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    args = TrainingArguments(
        output_dir=f"{out_dir}/checkpoints",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=epochs,
        max_steps=max_steps if max_steps > 0 else -1,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_steps=int(os.environ.get("SFT_WARMUP_STEPS", "2")),
        logging_steps=1,
        save_strategy="no",
        bf16=bf16,
        fp16=not bf16,
        optim="adamw_8bit",
        weight_decay=0.0,
        report_to="none",
        seed=0,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=Dataset.from_list(rows),
        data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100),
    )
    stats = trainer.train()
    print(json.dumps({"train_loss": stats.training_loss, "steps": stats.global_step}), flush=True)

    adapter_dir = f"{out_dir}/adapter"
    model.save_pretrained(adapter_dir)
    tok.save_pretrained(adapter_dir)
    with open(f"{out_dir}/train_log.json", "w") as f:
        json.dump(trainer.state.log_history, f, indent=1)
    import shutil

    shutil.make_archive(f"{out_dir}/adapter", "zip", adapter_dir)
    print(f"saved {adapter_dir}.zip", flush=True)


if __name__ == "__main__":
    main()
