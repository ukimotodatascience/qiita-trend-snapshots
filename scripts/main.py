import os
from pathlib import Path

import pandas as pd
import streamlit as st
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# --- 設定 ---
DEFAULT_CSV_PATH = "data/frequencies/summary/daily_words_all.csv"

# まずは repo 内のフォントを優先（あなたのコードを踏襲）
CANDIDATE_FONTS = [
    r"font/meiryo.ttc",
]


def find_font_path() -> str | None:
    for p in CANDIDATE_FONTS:
        if Path(p).exists():
            return p
    return None


@st.cache_data
def load_freq(
    csv_path: str, start_date: pd.Timestamp, end_date: pd.Timestamp
) -> dict[str, int]:
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]
    # 想定: date, word, count
    grouped = df.groupby("word", as_index=False)["count"].sum()
    freq = dict(zip(grouped["word"].astype(str), grouped["count"].astype(int)))
    return freq


@st.cache_data
def load_latest_date(csv_path: str) -> pd.Timestamp:
    df = pd.read_csv(csv_path, usecols=["date"])
    df["date"] = pd.to_datetime(df["date"])
    return df["date"].max()


def make_wordcloud(freq: dict[str, int], font_path: str) -> WordCloud:
    wc = WordCloud(
        width=1200,
        height=600,
        background_color="white",
        collocations=False,
        font_path=font_path,
    ).generate_from_frequencies(freq)
    return wc


# --- UI ---
# --- UI ---
st.set_page_config(page_title="WordCloud Viewer", layout="wide")
st.title("WordCloud（CSVの頻度から生成）")

period_options = {
    "直近1週間": 7,
    "直近1カ月": 30,
    "直近1年": 365,
}
period_label = st.radio("期間", list(period_options.keys()), horizontal=True)
period_days = period_options[period_label]

# CSVパスはコード内で固定（UIには表示しない）
csv_path = DEFAULT_CSV_PATH

font_path = find_font_path()
if font_path is None:
    st.error(
        "日本語フォントが見つかりませんでした。CANDIDATE_FONTS にフォントパスを追加してください。"
    )
    st.stop()

if not Path(csv_path).exists():
    st.error("内部CSVが見つかりません。データ生成処理を確認してください。")
    st.stop()

latest_date = load_latest_date(csv_path)
current_start = latest_date - pd.Timedelta(days=period_days - 1)
current_end = latest_date
prev_end = current_start - pd.Timedelta(days=1)
prev_start = prev_end - pd.Timedelta(days=period_days - 1)

current_freq = load_freq(csv_path, current_start, current_end)
prev_freq = load_freq(csv_path, prev_start, prev_end)

# WordCloud生成（固定値）
current_wc = WordCloud(
    width=1200,
    height=600,
    background_color="white",
    collocations=False,
    font_path=font_path,
    max_words=200,
).generate_from_frequencies(current_freq)

prev_wc = WordCloud(
    width=1200,
    height=600,
    background_color="white",
    collocations=False,
    font_path=font_path,
    max_words=200,
).generate_from_frequencies(prev_freq)

# 表示（matplotlib）
left_col, right_col = st.columns(2)
with left_col:
    st.subheader(f"現在の期間: {current_start.date()} 〜 {current_end.date()}")
    fig = plt.figure(figsize=(1200 / 200, 600 / 200))
    plt.imshow(current_wc, interpolation="bilinear")
    plt.axis("off")
    st.pyplot(fig, clear_figure=True)

with right_col:
    st.subheader(f"前の期間: {prev_start.date()} 〜 {prev_end.date()}")
    fig = plt.figure(figsize=(1200 / 200, 600 / 200))
    plt.imshow(prev_wc, interpolation="bilinear")
    plt.axis("off")
    st.pyplot(fig, clear_figure=True)

# ダウンロード（PNG）
img_bytes = current_wc.to_image()
import io

buf = io.BytesIO()
img_bytes.save(buf, format="PNG")
st.download_button(
    "PNGをダウンロード",
    data=buf.getvalue(),
    file_name="wordcloud.png",
    mime="image/png",
)
