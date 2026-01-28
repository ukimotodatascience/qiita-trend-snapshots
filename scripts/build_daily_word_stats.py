# scripts/build_daily_words.py
# Usage:
#   python scripts/build_daily_words.py --snapshot_dir data/snapshot --out_dir data/daily_words
# Optional:
#   python scripts/build_daily_words.py --stopwords data/meta/stopwords.txt

from __future__ import annotations

import argparse
import glob
import os
import re
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime
from typing import Dict, Iterable, List, Set, Tuple

import pandas as pd

# ----------------------------
# Tokenizer (Janome if available; otherwise regex fallback)
# ----------------------------
_JP_SEQ = re.compile(r"[ぁ-んァ-ヶ一-龠々ー]+")
_JP_SUBTOKEN = re.compile(r"[一-龠々]+|[ァ-ヶー]+|[ぁ-ん]{2,}")
_EN_SEQ = re.compile(r"[A-Za-z][A-Za-z0-9_\-+./]*")

# 英語/日本語の重み（お任せ指定のためデフォルトを設定）
EN_WEIGHT = 0.7
JP_WEIGHT = 1.0


def load_stopwords(path: str | None) -> Set[str]:
    """
    ストップワードファイルを読み込み、集合として返す。
    空行と「#」始まりの行は無視する。
    """
    if not path:
        return set()
    if not os.path.exists(path):
        raise FileNotFoundError(f"stopwords file not found: {path}")
    sw = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            w = line.strip()
            if not w or w.startswith("#"):
                continue
            sw.add(w)
    return sw


def try_janome():
    """
    Janomeが利用可能な場合は、JanomeのTokenizerを返し、そうでない場合はNoneを返す。
    """
    try:
        from janome.tokenizer import Tokenizer  # type: ignore

        return Tokenizer()
    except Exception:
        return None


def normalize_text(s: str) -> str:
    """
    文字列を正規化（NFKC）し、前後の空白を除去する。
    """
    # full-width -> half-width, etc.
    s = unicodedata.normalize("NFKC", str(s))
    return s.strip()


def _is_english_token(token: str) -> bool:
    return _EN_SEQ.fullmatch(token) is not None


def _split_jp_token(token: str) -> List[str]:
    return [m.group(0) for m in _JP_SUBTOKEN.finditer(token)]


def tokenize_title(
    title: str,
    stopwords: Set[str],
    min_len: int = 2,
) -> List[Tuple[str, float]]:
    """
    記事タイトルを単語リストに分割する。
    - Janome が使える場合は名詞のみ抽出
    - 使えない場合は正規表現で英語・日本語を簡易抽出
    - ストップワードが短すぎる語は除外
    """
    title = normalize_text(title)

    tk = try_janome()
    tokens: List[Tuple[str, float]] = []

    if tk is not None:
        # Janome: keep nouns only to avoid noisy particles
        for t in tk.tokenize(title):
            surface = t.surface
            pos = t.part_of_speech.split(",")[0] if t.part_of_speech else ""
            if pos != "名詞":
                continue
            w = normalize_text(surface).lower()
            if len(w) < min_len:
                continue
            if w in stopwords:
                continue
            weight = EN_WEIGHT if _is_english_token(w) else JP_WEIGHT
            tokens.append((w, weight))
        if tokens:
            return tokens  # Janome succeeded with some tokens

    # Fallback: EN words + JP sequences
    # EN: lower-cased
    for m in _EN_SEQ.finditer(title):
        w = m.group(0).lower()
        if len(w) < min_len or w in stopwords:
            continue
        tokens.append((w, EN_WEIGHT))

    # JP: keep sequences as tokens (coarse, but works without tokenizer)
    for m in _JP_SEQ.finditer(title):
        w = m.group(0)
        for sub in _split_jp_token(w):
            if len(sub) < min_len or sub in stopwords:
                continue
            tokens.append((sub, JP_WEIGHT))

    return tokens


# ----------------------------
# Snapshot -> Daily aggregation
# ----------------------------
def parse_date_from_feed_updated(feed_updated_at: str) -> str:
    """
    feed_update_at から日付を抽出する。
    """
    s = normalize_text(feed_updated_at)
    # datetime.fromisoformat supports '+09:00'
    dt = datetime.fromisoformat(s)
    return dt.date().isoformat()


def read_snapshot_csv(path: str) -> pd.DataFrame:
    """
    スナップショットのCSVを読み込み、必要な列を検証して返す。
    """
    df = pd.read_csv(path)
    required = {"feed_updated_at", "title", "url"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    # drop empty rows
    df = df.dropna(subset=["feed_updated_at", "title", "url"])
    df["feed_updated_at"] = df["feed_updated_at"].astype(str)
    df["title"] = df["title"].astype(str)
    df["url"] = df["url"].astype(str)
    return df


def build_daily_words(
    snapshot_dir: str,
    out_dir: str,
    stopwords_path: str | None = None,
    min_len: int = 2,
) -> None:
    """
    スナップショットのCSV群から日付別の単語出現統計を作成する。

    - 日付ごとにURLで記事を重複排除
    - タイトルから単語を抽出
    - 出現回数と記事数を集計してCSV出力
    """
    stopwords = load_stopwords(stopwords_path)

    paths = sorted(glob.glob(os.path.join(snapshot_dir, "*.csv")))
    if not paths:
        raise FileNotFoundError(f"No snapshot CSV found in: {snapshot_dir}")

    # date -> url -> title (dedupe within a date)
    daily_articles: Dict[str, Dict[str, str]] = defaultdict(dict)

    for p in paths:
        df = read_snapshot_csv(p)
        # feed_updated_at is same within file, but we handle row-wise safely
        for _, row in df.iterrows():
            date = parse_date_from_feed_updated(row["feed_updated_at"])
            url = row["url"].strip()
            title = row["title"].strip()
            if not url or not title:
                continue
            # dedupe by url; keep latest seen title (fine either way)
            daily_articles[date][url] = title

    os.makedirs(out_dir, exist_ok=True)

    for date, url_to_title in sorted(daily_articles.items()):
        # word -> total count across all titles
        word_count: Counter[str] = Counter()
        # word -> set(url) to compute article_count
        word_articles: Dict[str, Set[str]] = defaultdict(set)

        for url, title in url_to_title.items():
            tokens = tokenize_title(title, stopwords=stopwords, min_len=min_len)
            if not tokens:
                continue
            for w, weight in tokens:
                word_count[w] += weight
            for w in {w for w, _ in tokens}:
                word_articles[w].add(url)

        if not word_count:
            # still write empty file for the day if you want; here we skip
            continue

        out_path = os.path.join(out_dir, f"{date}.csv")
        out_df = pd.DataFrame(
            {
                "date": [date] * len(word_count),
                "word": list(word_count.keys()),
                "count": list(word_count.values()),
                "article_count": [len(word_articles[w]) for w in word_count.keys()],
            }
        ).sort_values(
            ["count", "article_count", "word"], ascending=[False, False, True]
        )

        out_df.to_csv(out_path, index=False, encoding="utf-8")
        print(
            f"[OK] wrote: {out_path} (articles={len(url_to_title)}, words={len(out_df)})"
        )


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


def build_monthly_words(
    daily_dir: str | None,
    daily_all_path: str | None,
    out_dir: str,
    out_all_path: str | None,
) -> None:
    """
    日次集計(date, word, count, article_count)を元に、月次集計を作る。
    - monthly_count = count の合計
    - monthly_article_count = article_count の合計（※延べ。URL重複排除はできない）
    - 月ごとに YYYY-MM.csv を出力
    - out_all_path が指定されれば monthly 全体を1つにまとめたCSVも出す
    """
    if (daily_dir is None) == (daily_all_path is None):
        raise ValueError("Specify exactly one of --daily_dir or --daily_all_path")

    df = pd.read_csv(daily_all_path)
    # date -> datetime
    df["date_dt"] = pd.to_datetime(df["date"], errors="raise")
    df["month"] = df["date_dt"].dt.to_period("M").astype(str)  # "YYYY-MM"

    # 月次集計
    mdf = (
        df.groupby(["month", "word"], as_index=False)
        .agg(
            count=("count", "sum"),
            article_count=("article_count", "sum"),
        )
        .sort_values(
            ["month", "count", "article_count", "word"],
            ascending=[True, False, False, True],
        )
    )

    os.makedirs(out_dir, exist_ok=True)

    # 月ごとに分割して保存
    for month, g in mdf.groupby("month", sort=True):
        out_path = os.path.join(out_dir, f"{month}.csv")
        out_df = g.copy()

    # 全月まとめ
    if out_all_path:
        os.makedirs(os.path.dirname(out_all_path) or ".", exist_ok=True)
        mdf.to_csv(out_all_path, index=False, encoding="utf-8")
        print(f"[OK] wrote: {out_all_path} (rows={len(mdf)})")


def build_yearly_words(
    daily_dir: str | None,
    daily_all_path: str | None,
    out_dir: str,
    out_all_path: str | None,
) -> None:
    """
    日次集計(date, word, count, article_count)を元に、年次集計を作る。
    - yearly_count = count の合計
    - yearly_article_count = article_count の合計（※延べ。URL重複排除はできない）
    - 年ごとに YYYY.csv を出力
    - out_all_path が指定されれば yearly 全体を1つにまとめたCSVも出す
    """
    if (daily_dir is None) == (daily_all_path is None):
        raise ValueError("Specify exactly one of --daily_dir or --daily_all_path")

    df = pd.read_csv(daily_all_path)

    df["date_dt"] = pd.to_datetime(df["date"], errors="raise")
    df["year"] = df["date_dt"].dt.year.astype(str)  # "YYYY"

    ydf = (
        df.groupby(["year", "word"], as_index=False)
        .agg(
            count=("count", "sum"),
            article_count=("article_count", "sum"),
        )
        .sort_values(
            ["year", "count", "article_count", "word"],
            ascending=[True, False, False, True],
        )
    )

    os.makedirs(out_dir, exist_ok=True)

    for year, g in ydf.groupby("year", sort=True):
        out_path = os.path.join(out_dir, f"{year}.csv")
        out_df = g.copy()

    if out_all_path:
        os.makedirs(os.path.dirname(out_all_path) or ".", exist_ok=True)
        ydf.to_csv(out_all_path, index=False, encoding="utf-8")
        print(f"[OK] wrote: {out_all_path} (rows={len(ydf)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--snapshot_dir",
        default="data/snapshots",
        help="directory containing snapshot CSVs",
    )
    ap.add_argument(
        "--daily_dir",
        default="data/frequencies",
        help="directory containing daily CSVs",
    )
    ap.add_argument(
        "--out_path",
        default="data/frequencies/summary/daily_words_all.csv",
        help="output CSV path",
    )
    args = ap.parse_args()

    # 1. スナップショットから日次単語CSVを生成（存在する場合）
    snapshot_paths = glob.glob(os.path.join(args.snapshot_dir, "*.csv"))
    if snapshot_paths:
        build_daily_words(
            snapshot_dir=args.snapshot_dir,
            out_dir=args.daily_dir,
        )

    # 2. 日次CSVを縦結合
    concat_daily_words(
        daily_dir=args.daily_dir,
        out_path=args.out_path,
    )

    build_monthly_words(
        daily_dir=None,
        daily_all_path=args.out_path,
        out_dir="data/frequencies/summary",
        out_all_path="data/frequencies/summary/monthly_words_all.csv",
    )

    build_yearly_words(
        daily_dir=None,
        daily_all_path=args.out_path,
        out_dir="data/frequencies/summary",
        out_all_path="data/frequencies/summary/yearly_words_all.csv",
    )


if __name__ == "__main__":
    main()
