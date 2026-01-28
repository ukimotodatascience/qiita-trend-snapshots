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
def load_freq(csv_path: str, days: int) -> dict[str, int]:
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    latest_date = df["date"].max()
    start_date = latest_date - pd.Timedelta(days=days - 1)
    df = df[df["date"] >= start_date]
    # 想定: date, word, count
    grouped = df.groupby("word", as_index=False)["count"].sum()
    freq = dict(zip(grouped["word"].astype(str), grouped["count"].astype(int)))
    return freq


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

freq = load_freq(csv_path, period_days)

# WordCloud生成（固定値）
wc = WordCloud(
    width=1200,
    height=600,
    background_color="white",
    collocations=False,
    font_path=font_path,
    max_words=200,
).generate_from_frequencies(freq)

# 表示（matplotlib）
fig = plt.figure(figsize=(1200 / 200, 600 / 200))
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
st.pyplot(fig, clear_figure=True)

# ダウンロード（PNG）
img_bytes = wc.to_image()
import io

buf = io.BytesIO()
img_bytes.save(buf, format="PNG")
st.download_button(
    "PNGをダウンロード",
    data=buf.getvalue(),
    file_name="wordcloud.png",
    mime="image/png",
)
