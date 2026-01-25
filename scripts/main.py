# import os
# from pathlib import Path
# import pandas as pd
# from wordcloud import WordCloud
# import matplotlib.pyplot as plt

# CSV_PATH = "data/daily_words/daily_words_all.csv"

# # Windowsならまずこのどれかが当たりやすい
# CANDIDATE_FONTS = [
#     r"font/meiryo.ttc",  # メイリオ
# ]

# FONT_PATH = next((p for p in CANDIDATE_FONTS if Path(p).exists()), None)
# if FONT_PATH is None:
#     raise FileNotFoundError(
#         "日本語フォントが見つかりませんでした。CANDIDATE_FONTS にフォントパスを追加してください。"
#     )

# df = pd.read_csv(CSV_PATH)
# freq = dict(zip(df["word"].astype(str), df["count"].astype(int)))

# wc = WordCloud(
#     width=1200,
#     height=600,
#     background_color="white",
#     collocations=False,
#     font_path=FONT_PATH,  # ← ここが重要
# ).generate_from_frequencies(freq)

# plt.figure(figsize=(12, 6))
# plt.imshow(wc, interpolation="bilinear")
# plt.axis("off")
# plt.tight_layout()
# plt.show()

# out_path = "wordcloud_2026-01-25.png"
# wc.to_file(out_path)
# print("saved:", os.path.abspath(out_path))
# print("font:", FONT_PATH)

import os
from pathlib import Path

import pandas as pd
import streamlit as st
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# --- 設定 ---
DEFAULT_CSV_PATH = "data/daily_words/daily_words_all.csv"

# まずは repo 内のフォントを優先（あなたのコードを踏襲）
CANDIDATE_FONTS = [
    r"font/meiryo.ttc",  # 例: リポジトリに同梱している場合
    # ここに他の候補も追加OK
    # r"font/NotoSansCJKjp-Regular.otf",
]


def find_font_path() -> str | None:
    for p in CANDIDATE_FONTS:
        if Path(p).exists():
            return p
    return None


@st.cache_data
def load_freq(csv_path: str) -> dict[str, int]:
    df = pd.read_csv(csv_path)
    # 想定: word, count
    freq = dict(zip(df["word"].astype(str), df["count"].astype(int)))
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
st.set_page_config(page_title="WordCloud Viewer", layout="wide")
st.title("WordCloud（CSVの頻度から生成）")

csv_path = st.text_input("CSV_PATH", DEFAULT_CSV_PATH)

font_path = find_font_path()
if font_path is None:
    st.error(
        "日本語フォントが見つかりませんでした。CANDIDATE_FONTS にフォントパスを追加してください。"
    )
    st.stop()

st.caption(f"font: `{font_path}`")

if not Path(csv_path).exists():
    st.error(f"CSVが見つかりません: {csv_path}")
    st.stop()

freq = load_freq(csv_path)

# オプション（必要なら）
with st.expander("表示オプション", expanded=False):
    max_words = st.slider("最大単語数", 50, 400, 200, 10)
    width = st.slider("幅", 600, 2000, 1200, 50)
    height = st.slider("高さ", 300, 1200, 600, 50)

# WordCloud生成
wc = WordCloud(
    width=width,
    height=height,
    background_color="white",
    collocations=False,
    font_path=font_path,
    max_words=max_words,
).generate_from_frequencies(freq)

# 表示（matplotlib）
fig = plt.figure(figsize=(width / 200, height / 200))
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
