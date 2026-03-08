from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator


HookTransform = Callable[[dict[str, Any]], dict[str, str] | None]


@dataclass
class PreparedRecord:
    input: str
    target: str
    left_context: str = ""

    def to_row(self) -> dict[str, str]:
        prompt = f"\uee02{self.left_context}\uee00{self.input}\uee01"
        return {
            "left_context": self.left_context,
            "input": self.input,
            "expected": self.target,
            "prompt": prompt,
        }


def load_hooks(module_name: str | None) -> HookTransform | None:
    if not module_name:
        return None

    module = importlib.import_module(module_name)

    if not hasattr(module, "transform_record"):
        raise ValueError(
            f"{module_name} must define transform_record(record: dict[str, Any]) -> dict[str, str] | None"
        )

    transform = getattr(module, "transform_record")
    if not callable(transform):
        raise ValueError(f"{module_name}.transform_record is not callable")

    return transform


def iter_hf_records(dataset: Iterable[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    for row in dataset:
        yield row


def iter_tsv_records(path: Path, with_context: bool) -> Iterator[dict[str, Any]]:
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

                left_context = fields[0]
                input_text = fields[1]
                target_text = fields[2]

                yield {
                    "left_context": left_context,
                    "input": input_text,
                    "target": target_text,
                }
            else:
                if len(fields) < 2:
                    raise ValueError(
                        f"Invalid line at {path}:{line_no}: expected 2+ columns, got {len(fields)}"
                    )

                input_text = fields[0]
                target_text = fields[1]

                yield {
                    "left_context": "",
                    "input": input_text,
                    "target": target_text,
                }


def normalize_record(record: dict[str, Any]) -> PreparedRecord | None:
    input_text = record.get("input")
    target_text = record.get("target")
    left_context = record.get("left_context", "")

    if input_text is None or target_text is None:
        return None

    input_text = str(input_text).strip()
    target_text = str(target_text).strip()
    left_context = "" if left_context is None else str(left_context).strip()

    if not input_text or not target_text:
        return None

    return PreparedRecord(
        left_context=left_context,
        input=input_text,
        target=target_text,
    )


def iter_prepared_records(
    source: Iterable[dict[str, Any]],
    hooks: HookTransform | None = None,
) -> Iterator[PreparedRecord]:
    for raw in source:
        transformed = hooks(raw) if hooks is not None else raw
        if transformed is None:
            continue

        prepared = normalize_record(transformed)
        if prepared is None:
            continue

        yield prepared


def read_prepared_records_from_path(path: Path, with_context: bool) -> list[dict[str, str]]:
    source = iter_tsv_records(path=path, with_context=with_context)
    return [prepared.to_row() for prepared in iter_prepared_records(source=source, hooks=None)]


def write_prepared_tsv(
    records: Iterable[PreparedRecord],
    output_path: Path,
    with_context: bool,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with output_path.open("w", encoding="utf-8") as f:
        for rec in records:
            if with_context:
                f.write(f"{rec.left_context}\t{rec.input}\t{rec.target}\n")
            else:
                f.write(f"{rec.input}\t{rec.target}\n")
            count += 1

    return count


def write_prepared_jsonl(
    records: Iterable[PreparedRecord],
    output_path: Path,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with output_path.open("w", encoding="utf-8") as f:
        for rec in records:
            obj = {
                "left_context": rec.left_context,
                "input": rec.input,
                "target": rec.target,
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            count += 1

    return count
