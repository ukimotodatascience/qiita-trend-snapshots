import csv
import os
from urllib.request import urlopen
import xml.etree.ElementTree as ET

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]   # scripts/ の1つ上
OUT_DIR = REPO_ROOT / "data" / "snapshots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FEED_URL = "https://qiita.com/popular-items/feed"

ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}


def sanitize_timestamp(ts: str) -> str:
    # ファイル名に使えない ":" を "-" に
    return ts.replace(":", "-")


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def main():
    xml_bytes = urlopen(FEED_URL).read()
    root = ET.fromstring(xml_bytes)

    # フィード全体の更新日時
    feed_updated_at = root.findtext(
        "atom:updated", default="", namespaces=ATOM_NS
    ).strip()

    if not feed_updated_at:
        raise RuntimeError("feed-level <updated> not found")

    safe_ts = sanitize_timestamp(feed_updated_at)
    filename = f"qiita_popular_{safe_ts}.csv"
    out_path = os.path.join(OUT_DIR, filename)

    # すでに同じ更新回のファイルがあれば何もしない
    if os.path.exists(out_path):
        print(f"Snapshot already exists: {out_path}")
        return

    entries = root.findall("atom:entry", ATOM_NS)

    ensure_dir(OUT_DIR)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["feed_updated_at", "title", "url"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

        for e in entries:
            title = e.findtext("atom:title", default="", namespaces=ATOM_NS).strip()
            link_el = e.find("atom:link", ATOM_NS)
            url = (link_el.attrib.get("href") if link_el is not None else "") or ""
            url = url.strip()

            if not title or not url:
                continue

            w.writerow(
                {
                    "feed_updated_at": feed_updated_at,
                    "title": title,
                    "url": url,
                }
            )

    print(f"Saved snapshot: {out_path} ({len(entries)} entries)")


if __name__ == "__main__":
    main()
