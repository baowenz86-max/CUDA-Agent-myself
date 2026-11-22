"""Offline deterministic SASS-to-CUDA generation, one GPU per process."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import time


def extract_cuda(text):
    blocks = re.findall(r"```[^\n]*\n(.*?)```", text, re.S)
    if len(blocks) == 1:
        return blocks[0].strip() + "\n"
    if not blocks and re.search(r"#\s*include|__global__", text):
        return text.strip() + "\n"
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--adapter", type=Path)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--limit", type=int, default=5)
    p.add_argument("--context", type=int, default=32768)
    p.add_argument("--max-new-tokens", type=int, default=8192)
    args = p.parse_args()
    if min(args.limit, args.context, args.max_new_tokens) <= 0:
        p.error("limits must be positive")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("Generation requires a GPU")
    # Only the user message is passed to the model. Never use the CUDA answer.
    samples = []
    with args.dataset.open() as f:
        for line in f:
            row = json.loads(line)
            user = row["messages"][0]
            if user["role"] != "user":
                raise ValueError("Expected user message")
            samples.append({"id": row["id"], "messages": [user]})
    samples.sort(key=lambda x: x["id"])
    samples = samples[:args.limit]
    if not samples:
        raise ValueError("Empty dataset")
    config = {"model": str(args.model.resolve()),
              "adapter": str(args.adapter.resolve()) if args.adapter else None,
              "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
              "context": args.context, "max_new_tokens": args.max_new_tokens,
              "ids": [s["id"] for s in samples], "do_sample": False}
    args.output.mkdir(parents=True, exist_ok=True)
    config_path = args.output / "generation_config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Output has different generation settings; choose a new directory")
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    tokenizer = AutoTokenizer.from_pretrained(args.model.resolve(), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model.resolve(), local_files_only=True, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa")
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter.resolve(), local_files_only=True)
    model.to("cuda").eval()
    model.config.use_cache = True
    for i, sample in enumerate(samples, 1):
        task = args.output / sample["id"]
        task.mkdir(exist_ok=True)
        result_path = task / "result.json"
        if result_path.exists():
            print(f"[{i}/{len(samples)}] {sample['id']} cached", flush=True)
            continue
        rendered = tokenizer.apply_chat_template(
            sample["messages"], tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(rendered, add_special_tokens=False, return_tensors="pt").to("cuda")
        prompt_tokens = inputs["input_ids"].shape[1]
        budget = min(args.max_new_tokens, args.context - prompt_tokens)
        if budget <= 0:
            raise ValueError(f"No output budget for {sample['id']}")
        print(f"[{i}/{len(samples)}] {sample['id']} prompt={prompt_tokens} budget={budget}", flush=True)
        started = time.monotonic()
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=budget, do_sample=False,
                                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                                    use_cache=True)
        generated = output[0, prompt_tokens:].tolist()
        text = tokenizer.decode(generated, skip_special_tokens=True)
        (task / "response.txt").write_text(text, encoding="utf-8")
        cuda = extract_cuda(text)
        if cuda:
            (task / "custom_ops.cu").write_text(cuda, encoding="utf-8")
        eos = model.generation_config.eos_token_id
        eos = eos if isinstance(eos, list) else [eos]
        truncated = len(generated) >= budget and (not generated or generated[-1] not in eos)
        result = {"id": sample["id"], "prompt_tokens": prompt_tokens,
                  "generated_tokens": len(generated), "output_budget": budget,
                  "seconds": time.monotonic() - started, "truncated": truncated,
                  "format_ok": cuda is not None, "status": "truncated" if truncated else
                  "generated" if cuda else "format_failed"}
        temporary = result_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(result, indent=2) + "\n")
        temporary.replace(result_path)
        print(json.dumps(result), flush=True)
        del output, inputs


if __name__ == "__main__":
    main()
