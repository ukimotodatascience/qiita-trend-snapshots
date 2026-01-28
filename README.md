# Qiita Trend Snapshots

Qiitaの「トレンド」記事を定期スナップショットし、
**タイトルから抽出したキーワードの頻度を集計・可視化**するためのリポジトリです。
日次・月次・年次の集計データを生成し、StreamlitでワードクラウドとTOP10を確認できます。

## ✨ 特徴

- **スナップショット収集**: Qiitaの人気記事をCSVとして保存
- **日本語対応のキーワード抽出**: SudachiPy/Janome/正規表現のフォールバックで名詞抽出
- **集計データの自動生成**: 日次→月次→年次の集計CSVを自動作成
- **可視化UI**: 期間別ワードクラウド + TOP10棒グラフ、比較表示、PNGダウンロード

## 🗂️ ディレクトリ構成

```
.
├─ data/
│  ├─ snapshots/                 # Qiita人気記事のスナップショットCSV
│  └─ frequencies/
│     ├─ YYYY-MM-DD.csv           # 日次キーワード集計
│     └─ summary/
│        ├─ daily_words_all.csv   # 日次CSVの結合
│        ├─ monthly_words_all.csv # 月次集計
│        └─ yearly_words_all.csv  # 年次集計
├─ font/                          # 日本語フォント
└─ scripts/
   ├─ fetch_qiita_popular.py      # 人気記事スナップショット取得
   ├─ build_daily_word_stats.py   # 集計CSV生成
   └─ main.py                     # Streamlitアプリ
```

## 🚀 使い方

### 1. 依存パッケージのインストール

```bash
pip install -r requirements.txt
```

### 2. スナップショットの取得

```bash
python scripts/fetch_qiita_popular.py
```

### 3. 集計CSVの作成

```bash
python scripts/build_daily_word_stats.py \
  --snapshot_dir data/snapshots \
  --daily_dir data/frequencies \
  --out_path data/frequencies/summary/daily_words_all.csv
```

### 4. ワードクラウドの確認

```bash
streamlit run scripts/main.py
```

## 🧠 集計ロジック概要

- **日次集計**: 同一日付のURLを重複排除し、タイトルから単語を抽出して頻度を集計
- **月次/年次集計**: 日次データを集約（延べ記事数として合算）
- **言語対応**: SudachiPy → Janome → 正規表現の順で抽出にフォールバック

## 🖼️ 生成イメージ

- ワードクラウド（指定期間/比較表示）
- 出現回数TOP10（横棒グラフ）
- PNGとしてダウンロード可能
<img width="1196" height="947" alt="image" src="https://github.com/user-attachments/assets/d3316790-4b82-45ce-a017-d8bfcf39b4d0" />

## ✅ こんな用途におすすめ

- Qiitaの技術トレンドを**定期観測**したい
- 記事タイトルから**キーワードの勢い**を把握したい
- **日本語ワードクラウド**を自動生成したい
