from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Callable


RE_WIDE_DIGITS = str.maketrans("０１２３４５６７８９", "0123456789")


def normalize_text(text: str) -> str:
    return unicodedata.normalize("NFKC", text).translate(RE_WIDE_DIGITS)


_DIGIT_KANA = {
    0: "ゼロ",
    1: "イチ",
    2: "ニ",
    3: "サン",
    4: "ヨン",
    5: "ゴ",
    6: "ロク",
    7: "ナナ",
    8: "ハチ",
    9: "キュウ",
}

_MONTH_KANA = {
    1: "イチガツ",
    2: "ニガツ",
    3: "サンガツ",
    4: "シガツ",
    5: "ゴガツ",
    6: "ロクガツ",
    7: "シチガツ",
    8: "ハチガツ",
    9: "クガツ",
    10: "ジュウガツ",
    11: "ジュウイチガツ",
    12: "ジュウニガツ",
}

_DAY_SPECIAL = {
    1: "ツイタチ",
    2: "フツカ",
    3: "ミッカ",
    4: "ヨッカ",
    5: "イツカ",
    6: "ムイカ",
    7: "ナノカ",
    8: "ヨウカ",
    9: "ココノカ",
    10: "トオカ",
    14: "ジュウヨッカ",
    20: "ハツカ",
    24: "ニジュウヨッカ",
}

_WEEKDAY_KANA = {
    "月": "ゲツ",
    "火": "カ",
    "水": "スイ",
    "木": "モク",
    "金": "キン",
    "土": "ド",
    "日": "ニチ",
}

_ERA_KANA = {
    "明治": "メイジ",
    "大正": "タイショウ",
    "昭和": "ショウワ",
    "平成": "ヘイセイ",
    "令和": "レイワ",
}

_ASCII_UNIT_KANA = {
    "km": "キロメートル",
    "cm": "センチメートル",
    "mm": "ミリメートル",
    "m": "メートル",
    "kg": "キログラム",
    "mg": "ミリグラム",
    "ml": "ミリリットル",
    "mb": "メガバイト",
    "gb": "ギガバイト",
    "tb": "テラバイト",
    "g": "グラム",
    "l": "リットル",
    "%": "パーセント",
}

# 長いもの優先
_ASCII_UNITS_SORTED = sorted(_ASCII_UNIT_KANA.keys(), key=len, reverse=True)


def _default_counter_reading(counter: str) -> Callable[[int], str]:
    return lambda n: int_to_kana(n) + counter


def read_people(n: int) -> str:
    if n == 1:
        return "ヒトリ"
    if n == 2:
        return "フタリ"
    return int_to_kana(n) + "ニン"


def read_age(n: int) -> str:
    if n == 20:
        return "ハタチ"
    return int_to_kana(n) + "サイ"


def read_hour(n: int) -> str:
    if n == 4:
        return "ヨジ"
    if n == 7:
        return "シチジ"
    if n == 9:
        return "クジ"
    return int_to_kana(n) + "ジ"


def read_minute(n: int) -> str:
    if n == 0:
        return "レイフン"

    exact = {
        1: "イップン",
        3: "サンプン",
        4: "ヨンプン",
        6: "ロップン",
        8: "ハップン",
        10: "ジュップン",
    }
    if n in exact:
        return exact[n]

    ones = n % 10
    if ones in {1, 3, 4, 6, 8}:
        base = int_to_kana(n - ones)
        tail = {
            1: "イップン",
            3: "サンプン",
            4: "ヨンプン",
            6: "ロップン",
            8: "ハップン",
        }[ones]
        return base + tail

    if n == 20:
        return "ニジップン"

    return int_to_kana(n) + "フン"


def read_day(n: int) -> str:
    if n in _DAY_SPECIAL:
        return _DAY_SPECIAL[n]
    return int_to_kana(n) + "ニチ"


def read_month(n: int) -> str | None:
    return _MONTH_KANA.get(n)


# かなり広めに最初から入れる
_COUNTER_READERS: dict[str, Callable[[int], str | None]] = {
    "年間": lambda n: int_to_kana(n) + "ネンカン",
    "周年": lambda n: int_to_kana(n) + "シュウネン",
    "年目": lambda n: int_to_kana(n) + "ネンメ",
    "か年": lambda n: int_to_kana(n) + "カネン",
    "ヶ年": lambda n: int_to_kana(n) + "カネン",
    "年": lambda n: int_to_kana(n) + "ネン",
    "か月": lambda n: int_to_kana(n) + "カゲツ",
    "ヶ月": lambda n: int_to_kana(n) + "カゲツ",
    "ヵ月": lambda n: int_to_kana(n) + "カゲツ",
    "ケ月": lambda n: int_to_kana(n) + "カゲツ",
    "月": lambda n: read_month(n),
    "日": lambda n: read_day(n),
    "時": lambda n: read_hour(n),
    "分": lambda n: read_minute(n),
    "秒": lambda n: int_to_kana(n) + "ビョウ",
    "人": lambda n: read_people(n),
    "名様": lambda n: int_to_kana(n) + "メイサマ",
    "名": lambda n: int_to_kana(n) + "メイ",
    "歳": lambda n: read_age(n),
    "才": lambda n: read_age(n),
    "円": lambda n: int_to_kana(n) + "エン",
    "回": lambda n: int_to_kana(n) + "カイ",
    "階": lambda n: int_to_kana(n) + "カイ",
    "章": lambda n: int_to_kana(n) + "ショウ",
    "話": lambda n: int_to_kana(n) + "ワ",
    "節": lambda n: int_to_kana(n) + "セツ",
    "倍": lambda n: int_to_kana(n) + "バイ",
    "株": lambda n: int_to_kana(n) + "カブ",
    "着": lambda n: int_to_kana(n) + "チャク",
    "個": lambda n: int_to_kana(n) + "コ",
    "件": lambda n: int_to_kana(n) + "ケン",
    "冊": lambda n: int_to_kana(n) + "サツ",
    "本": lambda n: int_to_kana(n) + "ホン",
    "台": lambda n: int_to_kana(n) + "ダイ",
    "箱": lambda n: int_to_kana(n) + "ハコ",
    "袋": lambda n: int_to_kana(n) + "フクロ",
    "杯": lambda n: int_to_kana(n) + "ハイ",
    "枚": lambda n: int_to_kana(n) + "マイ",
    "式": lambda n: int_to_kana(n) + "シキ",
    "点": lambda n: int_to_kana(n) + "テン",
    "行": lambda n: int_to_kana(n) + "ギョウ",
    "列": lambda n: int_to_kana(n) + "レツ",
    "校": lambda n: int_to_kana(n) + "コウ",
    "軒": lambda n: int_to_kana(n) + "ケン",
    "戸": lambda n: int_to_kana(n) + "コ",
    "局": lambda n: int_to_kana(n) + "キョク",
    "問": lambda n: int_to_kana(n) + "モン",
    "頁": lambda n: int_to_kana(n) + "ページ",
    "ページ": lambda n: int_to_kana(n) + "ページ",
    "ケース": lambda n: int_to_kana(n) + "ケース",
    "セット": lambda n: int_to_kana(n) + "セット",
    "cm": lambda n: int_to_kana(n) + "センチメートル",
    "mm": lambda n: int_to_kana(n) + "ミリメートル",
    "km": lambda n: int_to_kana(n) + "キロメートル",
    "kg": lambda n: int_to_kana(n) + "キログラム",
    "mg": lambda n: int_to_kana(n) + "ミリグラム",
    "ml": lambda n: int_to_kana(n) + "ミリリットル",
    "mb": lambda n: int_to_kana(n) + "メガバイト",
    "gb": lambda n: int_to_kana(n) + "ギガバイト",
    "tb": lambda n: int_to_kana(n) + "テラバイト",
    "g": lambda n: int_to_kana(n) + "グラム",
    "l": lambda n: int_to_kana(n) + "リットル",
    "%": lambda n: int_to_kana(n) + "パーセント",
}

_COUNTERS_SORTED = sorted(_COUNTER_READERS.keys(), key=len, reverse=True)

RE_YEAR_MONTH_DAY = re.compile(r"^(\d{4})[/-](\d{1,2})[/-](\d{1,2})")
RE_TIME_HM = re.compile(r"^(\d{1,2}):(\d{2})")
RE_DECIMAL = re.compile(r"^\d[\d,]*\.\d+")
RE_INTEGER = re.compile(r"^\d[\d,]*")


@dataclass
class NumericRuleResult:
    start: int
    end: int
    reading: str


def _parse_int(text: str) -> int | None:
    s = text.replace(",", "")
    if not s.isdigit():
        return None
    return int(s)


def _read_under_10000(x: int) -> str:
    out: list[str] = []

    thousands = x // 1000
    hundreds = (x % 1000) // 100
    tens = (x % 100) // 10
    ones = x % 10

    if thousands:
        if thousands == 1:
            out.append("セン")
        elif thousands == 3:
            out.append("サンゼン")
        elif thousands == 8:
            out.append("ハッセン")
        else:
            out.append(_DIGIT_KANA[thousands] + "セン")

    if hundreds:
        if hundreds == 1:
            out.append("ヒャク")
        elif hundreds == 3:
            out.append("サンビャク")
        elif hundreds == 6:
            out.append("ロッピャク")
        elif hundreds == 8:
            out.append("ハッピャク")
        else:
            out.append(_DIGIT_KANA[hundreds] + "ヒャク")

    if tens:
        if tens == 1:
            out.append("ジュウ")
        else:
            out.append(_DIGIT_KANA[tens] + "ジュウ")

    if ones:
        out.append(_DIGIT_KANA[ones])

    return "".join(out)


def int_to_kana(n: int) -> str:
    if n < 0:
        return "マイナス" + int_to_kana(-n)
    if n == 0:
        return "ゼロ"

    units = [
        (10**16, "ケイ"),
        (10**12, "チョウ"),
        (10**8, "オク"),
        (10**4, "マン"),
    ]

    parts: list[str] = []
    remaining = n

    for unit_value, unit_name in units:
        q = remaining // unit_value
        if q:
            parts.append(_read_under_10000(q) + unit_name)
            remaining %= unit_value

    if remaining:
        parts.append(_read_under_10000(remaining))

    return "".join(parts)


def digit_string_to_kana(text: str) -> str:
    return "".join(_DIGIT_KANA[int(ch)] for ch in text if ch.isdigit())


def read_letter_number(text: str) -> str:
    # ユーザー要望: アルファベットは変換しない
    return normalize_text(text)


def _match_counter_suffix(text: str, pos: int) -> str | None:
    lower_rest = text[pos:].lower()
    for counter in _COUNTERS_SORTED:
        if lower_rest.startswith(counter.lower()):
            return text[pos:pos + len(counter)]
    return None


def _read_counter(num: int, counter: str) -> str | None:
    fn = _COUNTER_READERS.get(counter)
    if fn is None:
        return None
    return fn(num)


def _read_decimal(text: str) -> str:
    normalized = text.replace(",", "")
    left, right = normalized.split(".", 1)
    left_read = int_to_kana(int(left)) if left else "ゼロ"
    right_read = digit_string_to_kana(right)
    return left_read + "テン" + right_read


def _scan_era_year(text: str, pos: int) -> tuple[str, int] | None:
    for era, era_read in _ERA_KANA.items():
        if not text.startswith(era, pos):
            continue
        i = pos + len(era)
        m = RE_INTEGER.match(text[i:])
        if not m:
            continue
        num_text = m.group(0)
        num = _parse_int(num_text)
        if num is None:
            continue
        j = i + len(num_text)
        if j < len(text) and text[j] == "年":
            return era_read + int_to_kana(num) + "ネン", j + 1
    return None


def _scan_date(text: str, pos: int) -> tuple[str, int] | None:
    m = RE_YEAR_MONTH_DAY.match(text[pos:])
    if not m:
        return None

    year = int(m.group(1))
    month = int(m.group(2))
    day = int(m.group(3))

    month_read = read_month(month)
    if month_read is None:
        return None

    reading = int_to_kana(year) + "ネン" + month_read + read_day(day)
    end = pos + len(m.group(0))

    if end < len(text) and text[end] in _WEEKDAY_KANA:
        reading += _WEEKDAY_KANA[text[end]]
        end += 1

    return reading, end


def _scan_time(text: str, pos: int) -> tuple[str, int] | None:
    m = RE_TIME_HM.match(text[pos:])
    if not m:
        return None
    hour = int(m.group(1))
    minute = int(m.group(2))
    return read_hour(hour) + read_minute(minute), pos + len(m.group(0))


def _scan_number(text: str, pos: int) -> tuple[str, int] | None:
    m = RE_DECIMAL.match(text[pos:])
    if m:
        num_text = m.group(0)
        end = pos + len(num_text)

        counter = _match_counter_suffix(text, end)
        if counter:
            # 小数+カウンタは基本そのまま接続
            counter_read = _COUNTER_READERS.get(counter)
            if counter_read is not None:
                return _read_decimal(num_text) + counter, end + len(counter)

        return _read_decimal(num_text), end

    m = RE_INTEGER.match(text[pos:])
    if not m:
        return None

    num_text = m.group(0)
    num = _parse_int(num_text)
    if num is None:
        return None

    end = pos + len(num_text)

    # 間の空白を許可
    space_end = end
    while space_end < len(text) and text[space_end].isspace():
        space_end += 1

    counter = _match_counter_suffix(text, space_end)
    if counter is not None:
        reading = _read_counter(num, counter)
        if reading is not None:
            return reading, space_end + len(counter)

    return int_to_kana(num), end


def find_numeric_rule_results(text: str) -> list[NumericRuleResult]:
    text = normalize_text(text)
    results: list[NumericRuleResult] = []

    i = 0
    n = len(text)

    while i < n:
        era = _scan_era_year(text, i)
        if era is not None:
            reading, end = era
            results.append(NumericRuleResult(i, end, reading))
            i = end
            continue

        date = _scan_date(text, i)
        if date is not None:
            reading, end = date
            results.append(NumericRuleResult(i, end, reading))
            i = end
            continue

        tm = _scan_time(text, i)
        if tm is not None:
            reading, end = tm
            results.append(NumericRuleResult(i, end, reading))
            i = end
            continue

        num = _scan_number(text, i)
        if num is not None:
            reading, end = num
            results.append(NumericRuleResult(i, end, reading))
            i = end
            continue

        i += 1

    return results


def replace_numeric_expressions_with_placeholders(text: str) -> tuple[str, dict[str, str]]:
    text = normalize_text(text)
    rules = find_numeric_rule_results(text)
    if not rules:
        return text, {}

    parts: list[str] = []
    mapping: dict[str, str] = {}
    cursor = 0

    for idx, rule in enumerate(rules):
        if rule.start < cursor:
            continue
        parts.append(text[cursor:rule.start])
        key = f"<NUM_{idx}>"
        parts.append(key)
        mapping[key] = rule.reading
        cursor = rule.end

    parts.append(text[cursor:])
    return "".join(parts), mapping
