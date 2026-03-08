from __future__ import annotations

import argparse

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

INPUT_START = "\uee00"
OUTPUT_START = "\uee01"
CONTEXT_START = "\uee02"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Greedy inference for kana-kanji conversion."
    )
    parser.add_argument("--lora_dir", type=str, required=True)
    parser.add_argument("--input", type=str, required=True, help="Yomi input")
    parser.add_argument(
        "--left_context",
        type=str,
        default="",
        help="Optional left context",
    )
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--repetition_penalty", type=float, default=1.2)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=3)
    return parser.parse_args()


def detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def pick_dtype(device: str) -> torch.dtype:
    if device == "cuda":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def build_prompt(yomi: str, left_context: str) -> str:
    if left_context:
        return f"{CONTEXT_START}{left_context}{INPUT_START}{yomi}{OUTPUT_START}"
    return f"{INPUT_START}{yomi}{OUTPUT_START}"


def get_num_embeddings_from_model(model) -> int:
    emb = model.get_input_embeddings()

    if hasattr(emb, "num_embeddings"):
        return emb.num_embeddings

    if hasattr(emb, "original_module") and hasattr(emb.original_module, "num_embeddings"):
        return emb.original_module.num_embeddings

    if hasattr(emb, "modules_to_save"):
        for _, module in emb.modules_to_save.items():
            if hasattr(module, "num_embeddings"):
                return module.num_embeddings

    raise AttributeError(
        f"Could not resolve num_embeddings from {type(emb).__name__}")


def extract_actual_text(full_text: str, tokenizer) -> str:
    if OUTPUT_START in full_text:
        actual = full_text.split(OUTPUT_START, 1)[1]
    else:
        actual = full_text

    if tokenizer.eos_token and tokenizer.eos_token in actual:
        actual = actual.split(tokenizer.eos_token, 1)[0]

    return actual.strip()


def main() -> None:
    args = parse_args()
    device = detect_device()
    dtype = pick_dtype(device)

    tokenizer = AutoTokenizer.from_pretrained(args.lora_dir, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    peft_config = PeftConfig.from_pretrained(args.lora_dir)
    base_model_name = peft_config.base_model_name_or_path

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=dtype,
    )
    base_model.resize_token_embeddings(len(tokenizer))

    model = PeftModel.from_pretrained(
        base_model,
        args.lora_dir,
    )

    model.eval()
    model.to(device)

    prompt = build_prompt(args.input, args.left_context)
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)

    print("base_model_name      =", base_model_name)
    print("tokenizer vocab size =", len(tokenizer))
    print("model vocab size     =", get_num_embeddings_from_model(model))
    print("prompt(repr)         =", repr(prompt))
    print("input_ids            =", encoded["input_ids"].tolist())
    print("input_ids max        =", int(encoded["input_ids"].max()))

    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **encoded,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            repetition_penalty=args.repetition_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=False)
    actual = extract_actual_text(full_text, tokenizer)

    print("--- DEBUG ---")
    print(f"raw(repr)    = {full_text!r}")
    print(f"actual       = {actual}")
    print("------------")


if __name__ == "__main__":
    main()
