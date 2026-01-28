import os
from pathlib import Path

import pandas as pd
import streamlit as st
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from matplotlib import font_manager

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
def load_date_range(csv_path: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    return df["date"].min(), df["date"].max()


def format_date_range(start: pd.Timestamp, end: pd.Timestamp) -> str:
    return f"{start.date()} 〜 {end.date()}"


def make_wordcloud(freq: dict[str, int], font_path: str) -> WordCloud:
    wc = WordCloud(
        width=700,
        height=350,
        background_color="white",
        collocations=False,
        font_path=font_path,
    ).generate_from_frequencies(freq)
    return wc


# --- UI ---
# --- UI ---
st.set_page_config(page_title="WordCloud Viewer", layout="wide")
st.title("Qiitaトレンド記事ワードクラウド")

period_options = {
    "直近1週間": 7,
    "直近1カ月": 30,
    "直近1年": 365,
}
period_label = st.radio("期間", list(period_options.keys()), horizontal=True)
period_days = period_options[period_label]

if "show_compare" not in st.session_state:
    st.session_state.show_compare = False


def enable_compare() -> None:
    st.session_state.show_compare = True


def reset_compare() -> None:
    st.session_state.show_compare = False


if st.session_state.show_compare:
    st.button("比較をリセット", on_click=reset_compare)
else:
    st.button("前の期間と比較する", on_click=enable_compare)

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

data_start, data_end = load_date_range(csv_path)
current_start = data_end - pd.Timedelta(days=period_days - 1)
current_end = data_end
prev_end = current_start - pd.Timedelta(days=1)
prev_start = prev_end - pd.Timedelta(days=period_days - 1)

current_freq = load_freq(csv_path, current_start, current_end)
prev_freq = load_freq(csv_path, prev_start, prev_end)

# WordCloud生成（固定値）
current_wc = None
if current_freq:
    current_wc = WordCloud(
        width=700,
        height=350,
        background_color="white",
        collocations=False,
        font_path=font_path,
        max_words=200,
    ).generate_from_frequencies(current_freq)

prev_wc = None
if st.session_state.show_compare and prev_freq:
    prev_wc = WordCloud(
        width=700,
        height=350,
        background_color="white",
        collocations=False,
        font_path=font_path,
        max_words=200,
    ).generate_from_frequencies(prev_freq)

# 表示（matplotlib）
st.caption(f"データ期間: {format_date_range(data_start, data_end)}")

if st.session_state.show_compare:
    left_col, right_col = st.columns(2)
    with left_col:
        if current_wc is None:
            st.info("現在の期間のデータがありません。")
        else:
            st.image(current_wc.to_array(), use_container_width=True)

    with right_col:
        if prev_wc is None:
            st.info("前の期間のデータがありません。")
        else:
            st.image(prev_wc.to_array(), use_container_width=True)
else:
    if current_wc is None:
        st.info("現在の期間のデータがありません。")
    else:
        st.image(current_wc.to_array(), use_container_width=True)

# ダウンロード（PNG）
if current_wc is not None:
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

# --- TOP10 横棒グラフ ---
if current_freq:
    st.subheader("出現回数TOP10")
    top10 = (
        pd.DataFrame(
            [{"word": word, "count": count} for word, count in current_freq.items()]
        )
        .sort_values("count", ascending=False)
        .head(10)
    )

    font_prop = font_manager.FontProperties(fname=font_path)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(top10["word"], top10["count"], color="#4C78A8")
    ax.invert_yaxis()
    ax.set_xlabel("出現回数", fontproperties=font_prop)
    ax.set_ylabel("ワード", fontproperties=font_prop)
    ax.tick_params(axis="y", labelsize=10)
    for label in ax.get_yticklabels():
        label.set_fontproperties(font_prop)
    st.pyplot(fig, use_container_width=True)
