from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)

INPUT_START = "\uee00"
OUTPUT_START = "\uee01"
CONTEXT_START = "\uee02"
SPECIAL_TOKENS = [INPUT_START, OUTPUT_START, CONTEXT_START]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LoRA fine-tuning for kana-kanji conversion with GPT-2 Japanese char model."
    )
    parser.add_argument("--train_tsv", type=Path, required=True)
    parser.add_argument("--valid_tsv", type=Path, required=True)
    parser.add_argument(
        "--base_model",
        type=str,
        default="ku-nlp/gpt2-small-japanese-char",
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--num_train_epochs", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--with_context",
        action="store_true",
        help="Expect TSV format: left_context<TAB>yomi<TAB>expected",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def pick_torch_dtype(device: str) -> torch.dtype:
    if device == "cuda":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def read_rows(path: Path, with_context: bool) -> list[dict[str, str]]:
    from scripts.dataset_pipeline import read_prepared_records_from_path

    return read_prepared_records_from_path(path=path, with_context=with_context)


def build_hf_dataset(rows: list[dict[str, str]]) -> Dataset:
    return Dataset.from_list(rows)


def load_tokenizer(base_model: str) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=False)
    added = tokenizer.add_special_tokens(
        {"additional_special_tokens": SPECIAL_TOKENS})
    print(f"Added {added} special tokens")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if tokenizer.eos_token is None:
        raise ValueError("Tokenizer eos_token is None")

    return tokenizer


def tokenize_and_mask(
    example: dict[str, str],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
) -> dict[str, Any]:
    prompt = example["prompt"]
    target = example["expected"] + tokenizer.eos_token
    text = prompt + target

    tokenized = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        padding=False,
        add_special_tokens=False,
    )

    prompt_ids = tokenizer(
        prompt,
        truncation=True,
        max_length=max_length,
        padding=False,
        add_special_tokens=False,
    )["input_ids"]

    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]

    labels = input_ids.copy()
    prompt_len = min(len(prompt_ids), len(labels))
    for i in range(prompt_len):
        labels[i] = -100

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


@dataclass
class KanaKanjiCollator:
    tokenizer: PreTrainedTokenizerBase

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            raise ValueError("pad_token_id is None")

        max_len = max(len(x["input_ids"]) for x in features)

        batch_input_ids: list[list[int]] = []
        batch_attention_mask: list[list[int]] = []
        batch_labels: list[list[int]] = []

        for feat in features:
            input_ids = feat["input_ids"]
            attention_mask = feat["attention_mask"]
            labels = feat["labels"]

            pad_len = max_len - len(input_ids)

            batch_input_ids.append(input_ids + [pad_id] * pad_len)
            batch_attention_mask.append(attention_mask + [0] * pad_len)
            batch_labels.append(labels + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }


def build_model(
    base_model: str,
    tokenizer: PreTrainedTokenizerBase,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
) -> AutoModelForCausalLM:
    model = AutoModelForCausalLM.from_pretrained(base_model)
    model.resize_token_embeddings(len(tokenizer))

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["c_attn", "c_proj", "c_fc"],
        modules_to_save=["wte", "lm_head"],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    train_rows = read_rows(args.train_tsv, args.with_context)
    valid_rows = read_rows(args.valid_tsv, args.with_context)

    if not train_rows:
        raise ValueError("Train TSV is empty")
    if not valid_rows:
        raise ValueError("Valid TSV is empty")

    print(f"Train rows: {len(train_rows)}")
    print(f"Valid rows: {len(valid_rows)}")

    tokenizer = load_tokenizer(args.base_model)

    train_ds = build_hf_dataset(train_rows)
    valid_ds = build_hf_dataset(valid_rows)

    train_ds = train_ds.map(
        lambda ex: tokenize_and_mask(ex, tokenizer, args.max_length),
        remove_columns=train_ds.column_names,
    )
    valid_ds = valid_ds.map(
        lambda ex: tokenize_and_mask(ex, tokenizer, args.max_length),
        remove_columns=valid_ds.column_names,
    )

    model = build_model(
        args.base_model,
        tokenizer,
        args.lora_r,
        args.lora_alpha,
        args.lora_dropout,
    )
    data_collator = KanaKanjiCollator(tokenizer)

    device = detect_device()
    dtype = pick_torch_dtype(device)
    fp16 = device == "cuda" and dtype == torch.float16
    bf16 = device == "cuda" and dtype == torch.bfloat16

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        overwrite_output_dir=True,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        evaluation_strategy="steps",
        save_strategy="steps",
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        fp16=fp16,
        bf16=bf16,
        dataloader_pin_memory=(device == "cuda"),
        remove_unused_columns=False,
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()
    metrics = trainer.evaluate()

    eval_loss = metrics.get("eval_loss")
    if eval_loss is not None:
        ppl = math.exp(eval_loss) if eval_loss < 20 else float("inf")
        print(f"eval_loss={eval_loss:.6f}")
        print(f"perplexity={ppl:.6f}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))

    print(f"Saved LoRA adapter and tokenizer to {args.output_dir}")


if __name__ == "__main__":
    main()
