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
