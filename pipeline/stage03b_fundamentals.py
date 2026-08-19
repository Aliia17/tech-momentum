"""Stage 03b — point-in-time fundamentals from SEC EDGAR + risk-free rate.

Fills the controls the paper uses that prices alone cannot give:

  BM   book equity / market cap. Book equity from us-gaap
       StockholdersEquity in actual 10-K/10-Q XBRL filings, taken
       point-in-time: at month t we use the latest value FILED on or
       before t (no look-ahead bias).
  ROE  trailing annual net income / book equity, same point-in-time rule.
  SIC  industry code (2-digit) for industry fixed effects.
  RF   1-month T-bill from the Ken French data library (US analogue of
       the paper's 1-month SHIBOR).

Sources: https://data.sec.gov XBRL/companyfacts + submissions APIs
(free, no key, requires User-Agent, ~10 req/s limit).

Outputs
  interim/fundamentals_monthly.parquet  month, ticker, be, ni, sic2
  interim/rf_monthly.parquet            month, rf (decimal monthly)

Run:  python pipeline/stage03b_fundamentals.py
"""

import io
import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

HEADERS = {"User-Agent": config.SEC_USER_AGENT}
EQUITY_TAGS = [
    "StockholdersEquity",
    "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
]
NI_TAG = "NetIncomeLoss"
FRENCH_URL = ("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/"
              "ftp/F-F_Research_Data_Factors_CSV.zip")


def get_json(url: str, retries: int = 3) -> dict | None:
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=60)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 404:
                return None
            time.sleep(2 * (attempt + 1))       # 429/403: back off
        except requests.RequestException:
            time.sleep(2 * (attempt + 1))
    return None


def extract_events(facts: dict) -> pd.DataFrame:
    """(filed, end, book equity, annual net income) events for one firm."""
    gaap = facts.get("facts", {}).get("us-gaap", {})
    rows = []
    for tag in EQUITY_TAGS:
        for item in gaap.get(tag, {}).get("units", {}).get("USD", []):
            if item.get("form") in ("10-K", "10-Q", "20-F") and item.get("val") is not None:
                rows.append({"filed": item["filed"], "end": item["end"],
                             "be": float(item["val"]), "ni": np.nan})
        if rows:
            break
    for item in gaap.get(NI_TAG, {}).get("units", {}).get("USD", []):
        if item.get("form") not in ("10-K", "20-F") or item.get("val") is None:
            continue
        start, end = item.get("start"), item.get("end")
        if not start or not end:
            continue
        days = (pd.Timestamp(end) - pd.Timestamp(start)).days
        if 300 <= days <= 400:                   # annual durations only
            rows.append({"filed": item["filed"], "end": end,
                         "be": np.nan, "ni": float(item["val"])})
    return pd.DataFrame(rows)


def main() -> None:
    # ---- ticker -> CIK for our patent universe
    patents = pd.read_parquet(config.DATA_INTERIM / "patents.parquet",
                              columns=["ticker"])
    tickers = sorted(patents["ticker"].unique())
    sec = json.loads((config.DATA_RAW / "company_tickers.json").read_text(encoding="utf-8"))
    cik_map = {v["ticker"]: int(v["cik_str"]) for v in sec.values()}
    universe = [(t, cik_map[t]) for t in tickers if t in cik_map]
    print(f"fetching EDGAR facts for {len(universe):,} firms ...")

    events, sics = [], []
    for n, (ticker, cik) in enumerate(universe, 1):
        facts = get_json(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json")
        if facts:
            ev = extract_events(facts)
            if not ev.empty:
                ev["ticker"] = ticker
                events.append(ev)
        sub = get_json(f"https://data.sec.gov/submissions/CIK{cik:010d}.json")
        if sub and sub.get("sic"):
            sics.append({"ticker": ticker, "sic2": str(sub["sic"])[:2]})
        if n % 100 == 0:
            print(f"  {n}/{len(universe)} ({len(events)} with facts)")
        time.sleep(0.12)                         # stay under SEC rate limit

    ev = pd.concat(events, ignore_index=True)
    ev["filed"] = pd.to_datetime(ev["filed"])
    ev["end"] = pd.to_datetime(ev["end"])
    sic = pd.DataFrame(sics)

    # ---- collapse to a monthly point-in-time panel
    print("building monthly point-in-time panel ...")
    months = pd.date_range(config.SIGNAL_START, config.RETURNS_END, freq="ME")
    panels = []
    for ticker, g in ev.groupby("ticker"):
        be = (g.dropna(subset=["be"]).sort_values(["filed", "end"])
              .drop_duplicates("filed", keep="last"))
        ni = (g.dropna(subset=["ni"]).sort_values(["filed", "end"])
              .drop_duplicates("filed", keep="last"))
        out = pd.DataFrame({"month": months})
        if not be.empty:
            out = pd.merge_asof(out, be[["filed", "be"]],
                                left_on="month", right_on="filed").drop(columns="filed")
        if not ni.empty:
            out = pd.merge_asof(out, ni[["filed", "ni"]],
                                left_on="month", right_on="filed").drop(columns="filed")
        out["ticker"] = ticker
        panels.append(out)
    monthly = pd.concat(panels, ignore_index=True).merge(sic, on="ticker", how="left")
    monthly.to_parquet(config.DATA_INTERIM / "fundamentals_monthly.parquet",
                       index=False)
    print(f"  saved fundamentals_monthly.parquet: {len(monthly):,} rows, "
          f"{monthly['ticker'].nunique():,} firms, "
          f"be coverage {monthly['be'].notna().mean():.0%}")

    # ---- risk-free rate (Ken French monthly factors, RF column, percent)
    print("downloading risk-free rate (Ken French library) ...")
    z = zipfile.ZipFile(io.BytesIO(requests.get(FRENCH_URL, timeout=120,
                                                headers=HEADERS).content))
    raw = z.read(z.namelist()[0]).decode("latin-1")
    lines = [ln for ln in raw.splitlines()]
    start = next(i for i, ln in enumerate(lines) if ln.strip()[:6].isdigit())
    end = next(i for i in range(start, len(lines))
               if not lines[i].strip() or not lines[i].strip()[:6].isdigit())
    ff = pd.read_csv(io.StringIO("\n".join(lines[start:end])), header=None,
                     names=["ym", "mktrf", "smb", "hml", "rf"])
    ff["month"] = pd.to_datetime(ff["ym"].astype(str), format="%Y%m") + pd.offsets.MonthEnd(0)
    ff["rf"] = ff["rf"].astype(float) / 100.0
    ff[["month", "rf"]].to_parquet(config.DATA_INTERIM / "rf_monthly.parquet",
                                   index=False)
    print(f"  saved rf_monthly.parquet: {len(ff)} months "
          f"({ff['month'].min():%Y-%m} .. {ff['month'].max():%Y-%m})")


if __name__ == "__main__":
    main()
