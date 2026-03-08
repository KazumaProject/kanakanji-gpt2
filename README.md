以下が、**checkpoint からの再開方法を含めて更新した README 全文**です。

````markdown
# kanakanji-gpt2

`ku-nlp/gpt2-small-japanese-char` をベースにした、最小のかな漢字変換 fine-tuning パイプラインです。

## 仕様

入力 TSV は次の形式です。

```tsv
ふるーとをふく	フルートを吹く
きしゃできしゃした	汽車で帰社した
```
````

文脈ありの場合は次の形式です。

```tsv
前の文脈	ふるーとをふく	フルートを吹く
前の文脈	きしゃできしゃした	汽車で帰社した
```

学習時にはこれを次の 1 系列へ変換します。

### 文脈なし

```text
\uee00ふるーとをふく\uee01フルートを吹く
\uee00きしゃできしゃした\uee01汽車で帰社した
```

### 文脈あり

```text
\uee02前の文脈\uee00ふるーとをふく\uee01フルートを吹く
\uee02前の文脈\uee00きしゃできしゃした\uee01汽車で帰社した
```

- `\uee00`: 入力開始
- `\uee01`: 出力開始
- `\uee02`: 文脈開始

損失は **出力開始以降だけ** にかかるようにしてあります。
つまり、入力側や文脈側は teacher forcing のために系列へ含めますが、loss は `expected` 側だけにかかります。

---

## セットアップ

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

---

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

文脈ありの場合は、各行を次の 3 列にします。

```tsv
left_context	yomi	expected
```

例:

```tsv
きのうはがっきてんにいった	ふるーとをふく	フルートを吹く
しゃないでしんぶんをよんだ	きしゃできしゃした	汽車で帰社した
```

---

## JSONL へ変換したい場合

```bash
python -m scripts.build_dataset --input data/train.tsv --output data/train.jsonl
python -m scripts.build_dataset --input data/valid.tsv --output data/valid.jsonl
```

---

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

---

## LoRA 学習

### 最小例

```bash
python -m scripts.train_lora --train_tsv data/train.tsv --valid_tsv data/valid.tsv --base_model ku-nlp/gpt2-small-japanese-char --output_dir out/gpt2-kanakanji-lora --num_train_epochs 10 --learning_rate 5e-4 --per_device_train_batch_size 8 --per_device_eval_batch_size 8 --gradient_accumulation_steps 1 --max_length 128
```

### zenz データで学習する例

```bash
python -m scripts.train_lora --train_tsv data/zenz_train.tsv --valid_tsv data/zenz_valid.tsv --base_model ku-nlp/gpt2-small-japanese-char --output_dir out/zenz-lora --num_train_epochs 3 --learning_rate 1e-4 --per_device_train_batch_size 8 --per_device_eval_batch_size 8 --gradient_accumulation_steps 1 --max_length 128 --with_context
```

```bash
python -m scripts.eval_greedy --lora_dir out/zenz-lora --input ふるーとをふく
```

```bash
python -m scripts.merge_lora --lora_dir out/zenz-lora --output_dir out/zenz-merged
```

### 主なオプション

- `--train_tsv`: 学習データ TSV
- `--valid_tsv`: 検証データ TSV
- `--base_model`: ベースモデル
- `--output_dir`: 出力先
- `--max_length`: 最大トークン長
- `--num_train_epochs`: エポック数
- `--learning_rate`: 学習率
- `--per_device_train_batch_size`: 学習バッチサイズ
- `--per_device_eval_batch_size`: 検証バッチサイズ
- `--gradient_accumulation_steps`: 勾配蓄積
- `--with_context`: `left_context<TAB>yomi<TAB>expected` 形式を使う
- `--resume_from_checkpoint`: checkpoint から学習再開する

---

## checkpoint 保存と再開

学習中、`Trainer` により `output_dir` の中へ自動で checkpoint が保存されます。

例:

```text
out/zenz-lora/
├── checkpoint-200/
├── checkpoint-400/
└── ...
```

これらの checkpoint から途中再開できます。

### 最新 checkpoint から再開

```bash
python -m scripts.train_lora --train_tsv data/zenz_train.tsv --valid_tsv data/zenz_valid.tsv --base_model ku-nlp/gpt2-small-japanese-char --output_dir out/zenz-lora --num_train_epochs 3 --learning_rate 1e-4 --per_device_train_batch_size 8 --per_device_eval_batch_size 8 --gradient_accumulation_steps 1 --max_length 128 --with_context --resume_from_checkpoint true
```

### 特定 checkpoint から再開

```bash
python -m scripts.train_lora --train_tsv data/zenz_train.tsv --valid_tsv data/zenz_valid.tsv --base_model ku-nlp/gpt2-small-japanese-char --output_dir out/zenz-lora --num_train_epochs 3 --learning_rate 1e-4 --per_device_train_batch_size 8 --per_device_eval_batch_size 8 --gradient_accumulation_steps 1 --max_length 128 --with_context --resume_from_checkpoint out/zenz-lora/checkpoint-200
```

### 注意

- checkpoint 再開時は、基本的に **同じ設定** を使ってください。
- `output_dir` を使い回して新規学習を始めると、既存 checkpoint に上書きしようとして warning が出ることがあります。
- 新規学習と再開学習を混同しないようにするのがおすすめです。

---

## greedy 推論

```bash
python -m scripts.eval_greedy --lora_dir out/gpt2-kanakanji-lora --input ふるーとをふく
```

文脈あり推論に対応している場合は、実装に応じて `left_context` も指定してください。

---

## LoRA をマージ

```bash
python -m scripts.merge_lora --lora_dir out/gpt2-kanakanji-lora --output_dir out/gpt2-kanakanji-merged
```

---

## build_dataset の拡張

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
        "yomi": record.get("input", ""),
        "expected": record.get("output", ""),
    }
```

---

## ディレクトリ例

```text
kanakanji-gpt2/
├── data/
│   ├── train.tsv
│   ├── valid.tsv
│   ├── zenz_train.tsv
│   └── zenz_valid.tsv
├── out/
│   └── zenz-lora/
├── scripts/
│   ├── build_dataset.py
│   ├── train_lora.py
│   ├── eval_greedy.py
│   └── merge_lora.py
├── pyproject.toml
└── README.md
```

---

## 補足

- 最初は **ひらがな入力で統一**するのがおすすめです。
- 推論も学習と同じ形式にしてください。
- 最初は文脈なしで十分です。
- まずは train の一部を再現できるところまで確認してください。
- 長さや表記ゆれが大きいデータを最初から大量投入するより、まず小さく検証したほうが安定します。
- fine-tuning すると、特定ドメインには強くなりますが、元の変換バランスが崩れることがあります。そのため valid を必ず分けて確認するのがおすすめです。
