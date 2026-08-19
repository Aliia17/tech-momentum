"""Stage 03 — monthly returns and price-based characteristics.

Downloads daily prices for every matched ticker from Yahoo Finance and
builds the month-end panel used by the signal and the tests:

  ret        month t return (close-to-close, month-end)
  mcap       market cap proxy at month-end (close * sharesOutstanding)
  vol        std of daily returns over trailing 20 trading days
  to         mean daily turnover (volume / shares) over trailing 20 days
  rev        1-month reversal = month t return (lagged in tests)

BM and ROE need point-in-time fundamentals (Compustat); Yahoo snapshots
are not point-in-time, so those controls are omitted — a limitation to
state clearly when presenting results.

Output: interim/returns.parquet  (month, ticker, ret, ln_mcap, vol, to, rev)

Run:  python pipeline/stage03_returns.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

BATCH = 100  # tickers per yfinance request


def main() -> None:
    patents = pd.read_parquet(config.DATA_INTERIM / "patents.parquet",
                              columns=["ticker"])
    tickers = sorted(patents["ticker"].unique())
    print(f"Downloading daily prices for {len(tickers):,} tickers ...")

    frames = []
    shares_map: dict[str, float] = {}
    for i in range(0, len(tickers), BATCH):
        batch = tickers[i:i + BATCH]
        px = yf.download(
            batch, start=config.PATENT_START, end=config.RETURNS_END,
            interval="1d", auto_adjust=True, progress=False, threads=True,
        )
        if px.empty:
            continue
        close = px["Close"] if isinstance(px.columns, pd.MultiIndex) else px[["Close"]]
        volume = px["Volume"] if isinstance(px.columns, pd.MultiIndex) else px[["Volume"]]
        frames.append((close, volume))
        print(f"  {min(i + BATCH, len(tickers))}/{len(tickers)}")
        time.sleep(1)  # be polite to Yahoo

    close = pd.concat([f[0] for f in frames], axis=1)
    volume = pd.concat([f[1] for f in frames], axis=1)
    close = close.loc[:, ~close.columns.duplicated()]
    volume = volume.loc[:, ~volume.columns.duplicated()]
    print(f"price matrix: {close.shape[0]} days x {close.shape[1]} tickers")

    # shares outstanding (current snapshot; used for mcap/turnover proxies)
    print("Fetching shares outstanding ...")
    for t in close.columns:
        try:
            shares_map[t] = yf.Ticker(t).fast_info.get("shares", np.nan)
        except Exception:
            shares_map[t] = np.nan
    shares = pd.Series(shares_map)

    daily_ret = close.pct_change()
    month_close = close.resample("ME").last()
    ret = month_close.pct_change()
    vol20 = daily_ret.rolling(20).std().resample("ME").last()
    turn = volume.div(shares, axis=1)
    to20 = turn.rolling(20).mean().resample("ME").last()
    mcap = month_close.mul(shares, axis=1)

    panel = pd.concat(
        {
            "ret": ret.stack(),
            "ln_mcap": np.log(mcap).stack(),
            "vol": vol20.stack(),
            "to": to20.stack(),
        },
        axis=1,
    ).reset_index()
    panel.columns = ["month", "ticker", "ret", "ln_mcap", "vol", "to"]
    panel["rev"] = panel["ret"]  # month-t return; lagged where used in tests
    panel = panel.dropna(subset=["ret"])
    panel.to_parquet(config.DATA_INTERIM / "returns.parquet", index=False)
    print(f"saved returns.parquet: {len(panel):,} firm-months, "
          f"{panel['ticker'].nunique():,} tickers")


if __name__ == "__main__":
    main()
