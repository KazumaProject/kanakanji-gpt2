from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from typing import Any

from sudachipy import Dictionary, SplitMode

from scripts.utils.japanese_number_reading import (
    replace_numeric_expressions_with_placeholders,
)

DROP_BRACKETS = set('()（）[]［］{}｛｝「」『』【】〈〉《》〔〕"“”‘’#✰✔.«*:・-:：;；/／!?！？,')
KEEP_PUNCT = set("、。ー〜～")
RE_SPACES = re.compile(r"\s+")
RE_PLACEHOLDER = re.compile(r"(<NUM_\d+>)")

RE_URLISH = re.compile(
    r"(https?://|www\.|/[A-Za-z0-9._/\-]+|[A-Za-z0-9._\-]+\.html?\b)",
    re.IGNORECASE,
)
RE_ASCII_COMMA_LIST = re.compile(
    r"[A-Za-z]{3,}(?:,[A-Za-z]{3,})+,?", re.IGNORECASE)
RE_BROKEN_END = re.compile(r"[:/,\-]\s*$")
RE_ODD_SLASH_TIME = re.compile(r"\d+/\d+\s+\d:\s*$")

# 英字はそのまま残す方針なので、英字があるだけでは落とさない。
# ただし URL 断片や明らかなゴミは落とす。
RE_GARBLED_PATH = re.compile(r"^[:/A-Za-z0-9._\-]+$")
RE_TOO_MANY_SYMBOLS = re.compile(r"[^\wぁ-んァ-ヶ一-龠ー、。,.!?！？・:：;；/%\s]")


def hira_to_kata(text: str) -> str:
    out: list[str] = []
    for ch in text:
        code = ord(ch)
        if 0x3041 <= code <= 0x3096:
            out.append(chr(code + 0x60))
        else:
            out.append(ch)
    return "".join(out)


def is_emoji_char(ch: str) -> bool:
    cp = ord(ch)
    return (
        0x1F300 <= cp <= 0x1FAFF
        or 0x2600 <= cp <= 0x27BF
        or 0x1F1E6 <= cp <= 0x1F1FF
        or cp in {0xFE0F, 0x200D}
    )


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)

    out: list[str] = []
    for ch in text:
        if ch in DROP_BRACKETS:
            continue
        if is_emoji_char(ch):
            continue

        cat = unicodedata.category(ch)
        if cat.startswith("C"):
            continue
        if cat.startswith("S"):
            continue

        out.append(ch)

    text = "".join(out)
    text = RE_SPACES.sub(" ", text).strip()
    return text


def is_probably_bad_example(text: str) -> bool:
    if not text:
        return True

    if len(text) < 2:
        return True

    if RE_URLISH.search(text):
        return True

    if RE_ASCII_COMMA_LIST.search(text):
        return True

    if RE_BROKEN_END.search(text):
        return True

    if RE_ODD_SLASH_TIME.search(text):
        return True

    if RE_GARBLED_PATH.fullmatch(text) and "/" in text:
        return True

    bad_symbol_count = len(RE_TOO_MANY_SYMBOLS.findall(text))
    if bad_symbol_count >= 4:
        return True

    return False


@lru_cache(maxsize=1)
def get_tokenizer():
    return Dictionary().create()


def _read_non_numeric_segment(text: str) -> str:
    tokenizer = get_tokenizer()
    morphemes = tokenizer.tokenize(text, SplitMode.C)
    readings: list[str] = []

    for m in morphemes:
        surface = m.surface()
        if not surface:
            continue

        pos = m.part_of_speech()

        if pos and pos[0] == "補助記号":
            if all(ch in KEEP_PUNCT or ch == ":" for ch in surface):
                readings.append(surface)
            continue

        if all(ch in KEEP_PUNCT or ch == ":" for ch in surface):
            readings.append(surface)
            continue

        # アルファベットは読み変換しない
        if all(ch.isascii() for ch in surface):
            readings.append(surface)
            continue

        if all(
            ("ぁ" <= ch <= "ん")
            or ("ァ" <= ch <= "ヶ")
            or ch in KEEP_PUNCT
            or ch in {"ー", "ォ", "ぉ", "ゔ", "ヴ"}
            for ch in surface
        ):
            readings.append(hira_to_kata(surface))
            continue

        reading = m.reading_form()
        if not reading or reading == "*":
            reading = surface

        reading = hira_to_kata(unicodedata.normalize("NFKC", reading))
        readings.append(reading)

    return "".join(readings)


def build_reading(text: str) -> str | None:
    placeholder_text, mapping = replace_numeric_expressions_with_placeholders(
        text)
    parts = RE_PLACEHOLDER.split(placeholder_text)

    readings: list[str] = []
    for part in parts:
        if not part:
            continue
        if part in mapping:
            readings.append(mapping[part])
            continue
        readings.append(_read_non_numeric_segment(part))

    result = "".join(readings).strip()
    return result or None


def transform_record(record: dict[str, Any]) -> dict[str, str] | None:
    raw_left_context = record.get("left_context")
    raw_output = record.get("output")

    if raw_output is None:
        return None

    output_text = clean_text(str(raw_output))
    if not output_text:
        return None

    if is_probably_bad_example(output_text):
        return None

    left_context = "" if raw_left_context is None else clean_text(
        str(raw_left_context))
    if left_context and is_probably_bad_example(left_context):
        left_context = ""

    rebuilt_input = build_reading(output_text)
    if not rebuilt_input:
        return None

    return {
        "left_context": left_context,
        "input": rebuilt_input,
        "target": output_text,
    }
