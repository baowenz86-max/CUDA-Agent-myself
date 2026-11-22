"""Offline LoRA SFT for the exported task-level SASS dataset.

Run with torchrun for DDP. Use --prepare-only before allocating GPUs.
"""

import argparse
import inspect
import json
import math
import os
from pathlib import Path


def encode_sample(tokenizer, messages, max_length):
    if [m["role"] for m in messages] != ["user", "assistant"]:
        raise ValueError("Expected exactly a user prompt and an assistant answer")
    prompt = tokenizer.apply_chat_template(
        messages[:1], tokenize=False, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    ids = tokenizer.encode(full, add_special_tokens=False)
    if ids[:len(prompt_ids)] != prompt_ids:
        raise ValueError("Chat template token boundary differs; refusing an incorrect loss mask")
    if len(ids) > max_length:
        raise ValueError(f"Sample has {len(ids)} tokens, exceeds {max_length}; no truncation")
    if len(ids) <= len(prompt_ids):
        raise ValueError("No supervised answer tokens")
    return {"input_ids": ids, "attention_mask": [1] * len(ids),
            "labels": [-100] * len(prompt_ids) + ids[len(prompt_ids):]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=32768)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--epochs", type=float, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--longest", type=int, help="Use N longest training examples for memory smoke test")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()

    import torch
    from datasets import Dataset
    from transformers import (AutoTokenizer, AutoModelForCausalLM, Trainer,
                              TrainingArguments, set_seed)
    from peft import LoraConfig, get_peft_model

    rank = int(os.environ.get("RANK", "0"))
    set_seed(2026)
    if args.output.exists() and any(args.output.iterdir()) and not args.resume:
        raise ValueError("Output is nonempty; choose a new directory or --resume CHECKPOINT")
    tokenizer = AutoTokenizer.from_pretrained(args.model.resolve(), local_files_only=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    def load(split):
        items = []
        with (args.data / f"{split}.jsonl").open() as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    encoded = encode_sample(tokenizer, row["messages"], args.max_length)
                    items.append((row["id"], encoded))
        if not items:
            raise ValueError(f"Empty {split} dataset")
        return items

    train_items = load("train")
    valid_items = load("valid")
    if args.longest:
        if args.longest <= 0:
            raise ValueError("--longest must be positive")
        train_items.sort(key=lambda x: len(x[1]["input_ids"]), reverse=True)
        train_items = train_items[:args.longest]
    report = {"train": len(train_items), "valid": len(valid_items),
              "max_train_tokens": max(len(x[1]["input_ids"]) for x in train_items),
              "train_supervised_tokens": sum(sum(v != -100 for v in x[1]["labels"]) for x in train_items),
              "longest_train_ids": [x[0] for x in sorted(train_items,
                  key=lambda x: len(x[1]["input_ids"]), reverse=True)[:5]],
              "loss_mask": "prompt/padding=-100; assistant including end marker supervised"}
    if rank == 0:
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if args.prepare_only:
        return
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Training requires CUDA with BF16 support")

    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    torch.cuda.reset_peak_memory_stats()
    model = AutoModelForCausalLM.from_pretrained(
        args.model.resolve(), local_files_only=True, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa")
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        task_type="CAUSAL_LM", r=32, lora_alpha=64, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"]))
    if rank == 0:
        model.print_trainable_parameters()

    def collate(features):
        width = max(len(f["input_ids"]) for f in features)
        batch = {}
        for key, pad in (("input_ids", tokenizer.pad_token_id), ("attention_mask", 0), ("labels", -100)):
            batch[key] = torch.tensor([f[key] + [pad] * (width-len(f[key])) for f in features])
        return batch

    smoke = args.max_steps > 0
    interval = max(1, math.ceil(len(train_items) / (world * args.grad_accum) / 2))
    config = TrainingArguments(
        output_dir=str(args.output), per_device_train_batch_size=1,
        per_device_eval_batch_size=1, gradient_accumulation_steps=args.grad_accum,
        learning_rate=1e-4, num_train_epochs=args.epochs, max_steps=args.max_steps,
        bf16=True, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False, optim="adamw_torch",
        warmup_ratio=0.05, weight_decay=0.01, max_grad_norm=1.0,
        logging_steps=1 if smoke else 5, report_to="none", seed=2026,
        eval_strategy="no" if smoke else "steps",
        save_strategy="steps", save_steps=args.max_steps if smoke else interval,
        eval_steps=interval, save_total_limit=2,
        load_best_model_at_end=not smoke, metric_for_best_model="eval_loss",
        greater_is_better=False, prediction_loss_only=True,
        dataloader_num_workers=0, ddp_timeout=3600,
    )
    kwargs = dict(model=model, args=config, data_collator=collate,
                  train_dataset=Dataset.from_list([x[1] for x in train_items]),
                  eval_dataset=Dataset.from_list([x[1] for x in valid_items]))
    # Transformers renamed tokenizer to processing_class; support both interfaces.
    name = "processing_class" if "processing_class" in inspect.signature(Trainer.__init__).parameters else "tokenizer"
    kwargs[name] = tokenizer
    trainer = Trainer(**kwargs)
    result = trainer.train(resume_from_checkpoint=str(args.resume) if args.resume else None)
    trainer.save_model(str(args.output / "adapter"))
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(args.output / "adapter")
        report.update(result.metrics)
        report["peak_gpu_allocated_gib_rank0"] = torch.cuda.max_memory_allocated() / 1024**3
        report["best_checkpoint"] = trainer.state.best_model_checkpoint
        report["arguments"] = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
        (args.output / "run_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
