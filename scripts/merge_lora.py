from __future__ import annotations

import argparse

from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge LoRA adapter into base model weights."
    )
    parser.add_argument("--lora_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.lora_dir, use_fast=False)
    model = AutoPeftModelForCausalLM.from_pretrained(args.lora_dir)

    merged = model.merge_and_unload()
    merged.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(f"Merged model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
