"""Stage 01 — download raw data.

Fetches PatentsView bulk table zips (resumable: skips files already present
and complete) and the SEC ticker/name mapping. This is the long stage: the
three tables are ~10 GB unzipped; expect 30-90 min depending on connection.

Run:  python pipeline/stage01_download.py
"""

import json
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config


def download(url: str, dest: Path, headers: dict | None = None) -> None:
    """Stream url to dest with a progress line; skip if size matches remote."""
    r = requests.head(url, headers=headers, allow_redirects=True, timeout=60)
    remote_size = int(r.headers.get("Content-Length", 0))
    if dest.exists() and remote_size and dest.stat().st_size == remote_size:
        print(f"  {dest.name}: already downloaded ({remote_size/1e6:.0f} MB), skipping")
        return
    print(f"  {dest.name}: downloading {remote_size/1e6:.0f} MB ...")
    with requests.get(url, headers=headers, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        done = 0
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                done += len(chunk)
                if remote_size and done % (200 << 20) < (1 << 20):
                    print(f"    {done/1e6:,.0f} / {remote_size/1e6:,.0f} MB")
    print(f"  {dest.name}: done ({dest.stat().st_size/1e6:,.0f} MB)")


def main() -> None:
    print("[1/2] PatentsView bulk tables (USPTO final release via Zenodo)")
    for table in config.PATENTSVIEW_TABLES:
        url = f"{config.PATENTSVIEW_BASE}/{table}.tsv.zip?download=1"
        download(url, config.DATA_RAW / f"{table}.tsv.zip")

    print("[2/2] SEC company tickers")
    resp = requests.get(
        config.SEC_TICKERS_URL,
        headers={"User-Agent": config.SEC_USER_AGENT},
        timeout=60,
    )
    resp.raise_for_status()
    out = config.DATA_RAW / "company_tickers.json"
    out.write_text(json.dumps(resp.json()), encoding="utf-8")
    print(f"  saved {out.name} ({len(resp.json()):,} companies)")


if __name__ == "__main__":
    main()
