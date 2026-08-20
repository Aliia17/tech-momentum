"""Stage 03c — CRSP monthly returns (gold-standard alternative to stage 03).

Reads the licensed CRSP Monthly Stock File export from
data/raw/crsp_monthly.csv (NOT committed to git) and produces a returns
panel with the same schema as stage 03, so downstream stages can swap
sources with --returns returns_crsp:

  ret      CRSP RET: total monthly return INCLUDING dividends (Yahoo close-
           to-close misses none of the survivors but all of the dead)
  ln_mcap  ln(|PRC| x SHROUT): true historical shares, not a snapshot
  vol      std of trailing 12 monthly returns (monthly file has no dailies)
  to       monthly volume / shares outstanding
  rev      month-t return

Filters: common shares (SHRCD 10/11), NYSE/AMEX/NASDAQ (EXCHCD 1/2/3).
Tickers are CRSP's historical per-month tickers; when several PERMNOs share
a ticker in a month, the largest market cap wins.

Output: interim/returns_crsp.parquet
Run:    python pipeline/stage03c_crsp_returns.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

SRC = config.DATA_RAW / "crsp_monthly.csv"
USECOLS = ["PERMNO", "date", "SHRCD", "EXCHCD", "TICKER", "PRC", "VOL",
           "RET", "SHROUT"]


def main() -> None:
    print(f"reading {SRC.name} (~800 MB) ...")
    df = pd.read_csv(SRC, usecols=USECOLS, dtype={"TICKER": str},
                     low_memory=False)
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] >= config.PATENT_START]

    df = df[df["SHRCD"].isin([10, 11]) & df["EXCHCD"].isin([1, 2, 3])]
    df = df.dropna(subset=["TICKER"])
    df["ret"] = pd.to_numeric(df["RET"], errors="coerce")  # 'C'/'B' codes -> NaN
    df = df.dropna(subset=["ret"])

    df["mcap"] = df["PRC"].abs() * df["SHROUT"] * 1000     # SHROUT in thousands
    df["ln_mcap"] = np.log(df["mcap"].where(df["mcap"] > 0))
    df["to"] = 100 * df["VOL"] / (df["SHROUT"] * 1000)      # VOL in 100s of shares
    df["month"] = df["date"] + pd.offsets.MonthEnd(0)

    # PERMNO is CRSP's permanent firm id; tickers change over time (FB->META).
    # Our patent links use CURRENT tickers, so label each PERMNO's whole
    # history with its LATEST ticker — Meta's FB-era months then correctly
    # attach to META. When two PERMNOs end on the same ticker, the one with
    # the larger final market cap wins.
    last = (df.sort_values("month").groupby("PERMNO").tail(1)
            [["PERMNO", "TICKER", "mcap"]]
            .sort_values("mcap")
            .drop_duplicates("TICKER", keep="last"))
    df = df.merge(last[["PERMNO", "TICKER"]].rename(columns={"TICKER": "ticker"}),
                  on="PERMNO", how="inner")
    df = (df.sort_values("mcap")
          .drop_duplicates(["month", "ticker"], keep="last"))

    df = df.sort_values(["ticker", "month"])
    df["vol"] = (df.groupby("ticker")["ret"]
                 .transform(lambda s: s.rolling(12, min_periods=6).std()))
    df["rev"] = df["ret"]

    out = df[["month", "ticker", "ret", "ln_mcap", "vol", "to", "rev"]]
    out.to_parquet(config.DATA_INTERIM / "returns_crsp.parquet", index=False)
    print(f"saved returns_crsp.parquet: {len(out):,} firm-months, "
          f"{out['ticker'].nunique():,} tickers, "
          f"{out['month'].min():%Y-%m} .. {out['month'].max():%Y-%m}")


if __name__ == "__main__":
    main()
