from __future__ import annotations

import argparse
import random
from pathlib import Path

from datasets import load_dataset

from scripts.dataset_pipeline import PreparedRecord, iter_prepared_records, load_hooks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build train/valid dataset from Hugging Face dataset with optional hooks."
    )
    parser.add_argument("--hf_dataset", type=str, required=True)
    parser.add_argument("--hf_split", type=str, default="train")
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--max_records", type=int, default=None)
    parser.add_argument("--valid_ratio", type=float, default=0.01)
    parser.add_argument("--train_output", type=Path, required=True)
    parser.add_argument("--valid_output", type=Path, required=True)
    parser.add_argument("--hooks", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--progress_every",
        type=int,
        default=1000,
        help="Print progress every N seen records.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not (0.0 < args.valid_ratio < 1.0):
        raise ValueError("--valid_ratio must be between 0 and 1")

    hooks = load_hooks(args.hooks)

    print("[1/6] loading dataset...")
    print(f"  dataset      : {args.hf_dataset}")
    print(f"  split        : {args.hf_split}")
    print(f"  streaming    : {args.streaming}")
    print(
        f"  max_records  : {args.max_records if args.max_records is not None else 'ALL'}")
    print(f"  valid_ratio  : {args.valid_ratio}")
    print(f"  hooks        : {args.hooks or 'none'}")
    print(f"  train_output : {args.train_output}")
    print(f"  valid_output : {args.valid_output}")

    dataset = load_dataset(
        args.hf_dataset,
        split=args.hf_split,
        streaming=args.streaming,
    )

    print("[2/6] dataset loaded")
    print("[3/6] preparing output files...")

    args.train_output.parent.mkdir(parents=True, exist_ok=True)
    args.valid_output.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)

    seen = 0
    accepted = 0
    skipped = 0
    train_count = 0
    valid_count = 0

    print("[4/6] transforming and writing records...")

    with (
        args.train_output.open("w", encoding="utf-8") as train_f,
        args.valid_output.open("w", encoding="utf-8") as valid_f,
    ):
        for prepared in iter_prepared_records(dataset, hooks=hooks):
            seen += 1

            if args.max_records is not None and accepted >= args.max_records:
                break

            if not isinstance(prepared, PreparedRecord):
                skipped += 1
                continue

            line = f"{prepared.left_context}\t{prepared.input}\t{prepared.target}\n"

            if rng.random() < args.valid_ratio:
                valid_f.write(line)
                valid_count += 1
            else:
                train_f.write(line)
                train_count += 1

            accepted += 1

            if accepted % args.progress_every == 0:
                print(
                    f"  progress: accepted={accepted:,} "
                    f"(train={train_count:,}, valid={valid_count:,})"
                )

        # hooks によって None が返るケースを正しく数えたい場合のため、
        # 元データを直接数えるのではなく、accepted ベースで進捗を見る。
        # ただし streaming 時は総数不明なのでこの形が扱いやすい。

    print("[5/6] completed")
    print(f"  accepted : {accepted:,}")
    print(f"  train    : {train_count:,}")
    print(f"  valid    : {valid_count:,}")
    print(f"  skipped  : {skipped:,}")

    if accepted == 0:
        raise ValueError(
            "No records were written. Check your hook or dataset columns.")

    print("[6/6] done")


if __name__ == "__main__":
    main()
