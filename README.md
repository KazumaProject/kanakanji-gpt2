# kanakanji-gpt2

`ku-nlp/gpt2-small-japanese-char` をベースにした、最小のかな漢字変換 fine-tuning パイプラインです。

## 仕様

入力 TSV は次の形式です。

```tsv
ふるーとをふく	フルートを吹く
きしゃできしゃした	汽車で帰社した
```

学習時にはこれを次の 1 系列へ変換します。

```text
\uee00ふるーとをふく\uee01フルートを吹く
\uee00きしゃできしゃした\uee01汽車で帰社した
```

- `\uee00`: 入力開始
- `\uee01`: 出力開始
- `\uee02`: 文脈開始（今回の最小構成では未使用）

損失は **出力開始以降だけ** にかかるようにしてあります。

## セットアップ

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

## 学習データ例

`data/train.tsv`

```tsv
ふるーとをふく	フルートを吹く
はーもにかをふく	ハーモニカを吹く
ふえをふく	笛を吹く
ぬのでふく	布で拭く
ふきんでふく	布巾で拭く
ぞうきんでふく	雑巾で拭く
きしゃのきしゃ	貴社の記者
きしゃのきしゃが	貴社の記者が
きしゃのきしゃと	貴社の記者と
きしゃのきしゃは	貴社の記者は
きしゃのきしゃを	貴社の記者を
きしゃがきしゃ	記者が汽車
きしゃがきしゃで	記者が汽車で
きしゃがきしゃに	記者が汽車に
きしゃできしゃ	汽車で帰社
きしゃできしゃした	汽車で帰社した
きしゃできしゃする	汽車で帰社する
```

`data/valid.tsv` も同じ形式です。

## JSONL へ変換したい場合

```bash
python -m scripts.build_dataset --input data/train.tsv --output data/train.jsonl
python -m scripts.build_dataset --input data/valid.tsv --output data/valid.jsonl
```

## zenz-v2.5 dataset を使う

`Miwa-Keita/zenz-v2.5-dataset` は `input`, `output`, `left_context` を持つので、そのまま学習用 TSV / JSONL に変換できます。
データセットは非常に大きいので、最初は `--max_records` で件数を絞って前処理するのがおすすめです。

### まず 10 万件だけ作る

```bash
python -m scripts.build_dataset --hf_dataset Miwa-Keita/zenz-v2.5-dataset --hf_split train --streaming --max_records 100000 --valid_ratio 0.01 --train_output data/zenz_train.tsv --valid_output data/zenz_valid.tsv --hooks scripts.hooks.zenz_default
```

### 全件を対象に train / valid を作る

```bash
python -m scripts.build_dataset --hf_dataset Miwa-Keita/zenz-v2.5-dataset --hf_split train --streaming --valid_ratio 0.01 --train_output data/zenz_all_train.tsv --valid_output data/zenz_all_valid.tsv --hooks scripts.hooks.zenz_default
```

`--max_records` を省略すると、フィルター通過後の **全件** を順次書き出します。

### 学習

```bash
python -m scripts.train_lora --train_tsv data/zenz_train.tsv --valid_tsv data/zenz_valid.tsv --base_model ku-nlp/gpt2-small-japanese-char --output_dir out/zenz-lora --num_train_epochs 3 --learning_rate 1e-4 --per_device_train_batch_size 8 --per_device_eval_batch_size 8 --gradient_accumulation_steps 1 --max_length 128 --with_context
```

## 将来の拡張

`build_dataset.py` は `--hooks` で前処理モジュールを差し替えられるようにしてあります。

- `filter_record(record) -> bool` で除外条件を追加
- `transform_record(record) -> dict | None` で読みの変更や正規化を追加

例:

```python
def filter_record(record):
    return len(record.get("input", "")) <= 40


def transform_record(record):
    return {
        "left_context": record.get("left_context") or "",
        "yomi": record.get("input", "").replace("ヴ", "ブ"),
        "expected": record.get("output", ""),
    }
```

## LoRA 学習

```bash
python -m scripts.train_lora --train_tsv data/train.tsv --valid_tsv data/valid.tsv --base_model ku-nlp/gpt2-small-japanese-char --output_dir out/gpt2-kanakanji-lora --num_train_epochs 10 --learning_rate 5e-4 --per_device_train_batch_size 8 --per_device_eval_batch_size 8 --gradient_accumulation_steps 1 --max_length 128
```

## greedy 推論

```bash
python -m scripts.eval_greedy --lora_dir out/gpt2-kanakanji-lora --input ふるーとをふく
```

## LoRA をマージ

```bash
python -m scripts.merge_lora --lora_dir out/gpt2-kanakanji-lora --output_dir out/gpt2-kanakanji-merged
```

## 補足

- 最初は **ひらがな入力で統一**するのがおすすめです。
- 推論も学習と同じ形式にしてください。
- 最初は文脈なしで十分です。
- まずは train の一部を再現できるところまで確認してください。
