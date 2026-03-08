from __future__ import annotations

import argparse
import json
from pathlib import Path

INPUT_START = "\uee00"
OUTPUT_START = "\uee01"
CONTEXT_START = "\uee02"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert yomi<TAB>expected TSV into JSONL for kana-kanji training."
    )
    parser.add_argument("--input", type=Path, required=True, help="Input TSV path")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL path")
    parser.add_argument(
        "--with_context",
        action="store_true",
        help="Expect TSV format: left_context<TAB>yomi<TAB>expected",
    )
    return parser.parse_args()


def read_tsv(path: Path, with_context: bool) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    with path.open("r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.rstrip("\n")
            if not line.strip():
                continue

            fields = line.split("\t")

            if with_context:
                if len(fields) < 3:
                    raise ValueError(
                        f"Invalid line at {path}:{line_no}: expected 3+ columns, got {len(fields)}"
                    )
                left_context = fields[0].strip()
                yomi = fields[1].strip()
                expected = fields[2].strip()
            else:
                if len(fields) < 2:
                    raise ValueError(
                        f"Invalid line at {path}:{line_no}: expected 2+ columns, got {len(fields)}"
                    )
                left_context = ""
                yomi = fields[0].strip()
                expected = fields[1].strip()

            if not yomi or not expected:
                continue

            prompt = (
                f"{CONTEXT_START}{left_context}{INPUT_START}{yomi}{OUTPUT_START}"
                if left_context
                else f"{INPUT_START}{yomi}{OUTPUT_START}"
            )
            text = f"{prompt}{expected}"

            rows.append(
                {
                    "left_context": left_context,
                    "yomi": yomi,
                    "expected": expected,
                    "prompt": prompt,
                    "text": text,
                }
            )

    return rows


def main() -> None:
    args = parse_args()
    rows = read_tsv(args.input, args.with_context)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
