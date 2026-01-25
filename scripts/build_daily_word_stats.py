# # scripts/build_daily_words.py
# # Usage:
# #   python scripts/build_daily_words.py --snapshot_dir data/snapshot --out_dir data/daily_words
# # Optional:
# #   python scripts/build_daily_words.py --stopwords data/meta/stopwords.txt

# from __future__ import annotations

# import argparse
# import glob
# import os
# import re
# import unicodedata
# from collections import Counter, defaultdict
# from datetime import datetime
# from typing import Dict, Iterable, List, Set, Tuple

# import pandas as pd


# # ----------------------------
# # Tokenizer (Janome if available; otherwise regex fallback)
# # ----------------------------
# _JP_SEQ = re.compile(r"[ぁ-んァ-ヶ一-龠]+")
# _EN_SEQ = re.compile(r"[A-Za-z][A-Za-z0-9_\-+./]*")


# def load_stopwords(path: str | None) -> Set[str]:
#     """
#     ストップワードファイルを読み込み、集合として返す。
#     空行と「#」始まりの行は無視する。
#     """
#     if not path:
#         return set()
#     if not os.path.exists(path):
#         raise FileNotFoundError(f"stopwords file not found: {path}")
#     sw = set()
#     with open(path, "r", encoding="utf-8") as f:
#         for line in f:
#             w = line.strip()
#             if not w or w.startswith("#"):
#                 continue
#             sw.add(w)
#     return sw


# def try_janome():
#     """
#     Janomeが利用可能な場合は、JanomeのTokenizerを返し、そうでない場合はNoneを返す。
#     """
#     try:
#         from janome.tokenizer import Tokenizer  # type: ignore
#         return Tokenizer()
#     except Exception:
#         return None


# def normalize_text(s: str) -> str:
#     """
#     文字列を正規化（NFKC）し、前後の空白を除去する。
#     """
#     # full-width -> half-width, etc.
#     s = unicodedata.normalize("NFKC", str(s))
#     return s.strip()


# def tokenize_title(title: str, stopwords: Set[str], min_len: int = 2) -> List[str]:
#     """
#     記事タイトルを単語リストに分割する。
#     - Janome が使える場合は名詞のみ抽出
#     - 使えない場合は正規表現で英語・日本語を簡易抽出
#     - ストップワードが短すぎる語は除外
#     """
#     title = normalize_text(title)

#     tk = try_janome()
#     tokens: List[str] = []

#     if tk is not None:
#         # Janome: keep nouns only to avoid noisy particles
#         for t in tk.tokenize(title):
#             surface = t.surface
#             pos = t.part_of_speech.split(",")[0] if t.part_of_speech else ""
#             if pos != "名詞":
#                 continue
#             w = normalize_text(surface).lower()
#             if len(w) < min_len:
#                 continue
#             if w in stopwords:
#                 continue
#             tokens.append(w)
#         if tokens:
#             return tokens  # Janome succeeded with some tokens

#     # Fallback: EN words + JP sequences
#     # EN: lower-cased
#     for m in _EN_SEQ.finditer(title):
#         w = m.group(0).lower()
#         if len(w) < min_len or w in stopwords:
#             continue
#         tokens.append(w)

#     # JP: keep sequences as tokens (coarse, but works without tokenizer)
#     for m in _JP_SEQ.finditer(title):
#         w = m.group(0)
#         if len(w) < min_len or w in stopwords:
#             continue
#         tokens.append(w)

#     return tokens


# # ----------------------------
# # Snapshot -> Daily aggregation
# # ----------------------------
# def parse_date_from_feed_updated(feed_updated_at: str) -> str:
#     """
#     feed_update_at から日付を抽出する。
#     """
#     s = normalize_text(feed_updated_at)
#     # datetime.fromisoformat supports '+09:00'
#     dt = datetime.fromisoformat(s)
#     return dt.date().isoformat()


# def read_snapshot_csv(path: str) -> pd.DataFrame:
#     """
#     スナップショットのCSVを読み込み、必要な列を検証して返す。
#     """
#     df = pd.read_csv(path)
#     required = {"feed_updated_at", "title", "url"}
#     missing = required - set(df.columns)
#     if missing:
#         raise ValueError(f"{path} missing columns: {missing}")
#     # drop empty rows
#     df = df.dropna(subset=["feed_updated_at", "title", "url"])
#     df["feed_updated_at"] = df["feed_updated_at"].astype(str)
#     df["title"] = df["title"].astype(str)
#     df["url"] = df["url"].astype(str)
#     return df


# def build_daily_words(
#     snapshot_dir: str,
#     out_dir: str,
#     stopwords_path: str | None = None,
#     min_len: int = 2,
# ) -> None:
#     """
#     スナップショットのCSV群から日付別の単語出現統計を作成する。

#     - 日付ごとにURLで記事を重複排除
#     - タイトルから単語を抽出
#     - 出現回数と記事数を集計してCSV出力
#     """
#     stopwords = load_stopwords(stopwords_path)

#     paths = sorted(glob.glob(os.path.join(snapshot_dir, "*.csv")))
#     if not paths:
#         raise FileNotFoundError(f"No snapshot CSV found in: {snapshot_dir}")

#     # date -> url -> title (dedupe within a date)
#     daily_articles: Dict[str, Dict[str, str]] = defaultdict(dict)

#     for p in paths:
#         df = read_snapshot_csv(p)
#         # feed_updated_at is same within file, but we handle row-wise safely
#         for _, row in df.iterrows():
#             date = parse_date_from_feed_updated(row["feed_updated_at"])
#             url = row["url"].strip()
#             title = row["title"].strip()
#             if not url or not title:
#                 continue
#             # dedupe by url; keep latest seen title (fine either way)
#             daily_articles[date][url] = title

#     os.makedirs(out_dir, exist_ok=True)

#     for date, url_to_title in sorted(daily_articles.items()):
#         # word -> total count across all titles
#         word_count: Counter[str] = Counter()
#         # word -> set(url) to compute article_count
#         word_articles: Dict[str, Set[str]] = defaultdict(set)

#         for url, title in url_to_title.items():
#             tokens = tokenize_title(title, stopwords=stopwords, min_len=min_len)
#             if not tokens:
#                 continue
#             word_count.update(tokens)
#             for w in set(tokens):
#                 word_articles[w].add(url)

#         if not word_count:
#             # still write empty file for the day if you want; here we skip
#             continue

#         out_path = os.path.join(out_dir, f"{date}.csv")
#         out_df = pd.DataFrame(
#             {
#                 "date": [date] * len(word_count),
#                 "word": list(word_count.keys()),
#                 "count": list(word_count.values()),
#                 "article_count": [len(word_articles[w]) for w in word_count.keys()],
#             }
#         ).sort_values(["count", "article_count", "word"], ascending=[False, False, True])

#         out_df.to_csv(out_path, index=False, encoding="utf-8")
#         print(f"[OK] wrote: {out_path} (articles={len(url_to_title)}, words={len(out_df)})")


# def main():
#     """
#     コマンドライン引数を解析し、日次単語集計を実行する
#     """
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--snapshot_dir", default="data/snapshots", help="directory containing snapshot CSVs")
#     ap.add_argument("--out_dir", default="data/daily_words", help="output directory for daily aggregated CSVs")
#     ap.add_argument("--stopwords", default=None, help="path to stopwords txt (one word per line)")
#     ap.add_argument("--min_len", type=int, default=2, help="minimum token length")
#     args = ap.parse_args()

#     build_daily_words(
#         snapshot_dir=args.snapshot_dir,
#         out_dir=args.out_dir,
#         stopwords_path=args.stopwords,
#         min_len=args.min_len,
#     )


# if __name__ == "__main__":
#     main()

# scripts/concat_daily_words.py
# Usage:
#   python scripts/concat_daily_words.py \
#     --daily_dir data/daily_words \
#     --out_path data/daily_words_all.csv

from __future__ import annotations

import argparse
import glob
import os

import pandas as pd


def concat_daily_words(daily_dir: str, out_path: str) -> None:
    """
    daily_words 配下の CSV をすべて縦結合して 1 つの CSV にする
    """
    paths = sorted(glob.glob(os.path.join(daily_dir, "*.csv")))
    if not paths:
        raise FileNotFoundError(f"No CSV files found in: {daily_dir}")

    dfs = []
    for p in paths:
        df = pd.read_csv(p)

        required = {"date", "word", "count", "article_count"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{p} missing columns: {missing}")

        dfs.append(df)

    all_df = pd.concat(dfs, ignore_index=True)

    # date を文字列として保証（将来の groupby 用）
    all_df["date"] = all_df["date"].astype(str)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    all_df.to_csv(out_path, index=False, encoding="utf-8")

    print(f"[OK] wrote: {out_path} (rows={len(all_df)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--daily_dir", default="data/daily_words", help="directory containing daily CSVs")
    ap.add_argument("--out_path", default="data/daily_words/daily_words_all.csv", help="output CSV path")
    args = ap.parse_args()

    concat_daily_words(
        daily_dir=args.daily_dir,
        out_path=args.out_path,
    )


if __name__ == "__main__":
    main()
