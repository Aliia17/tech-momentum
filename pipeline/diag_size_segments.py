"""Diagnostic — TECHMOM (bge-large) by size segment: S&P-proxy vs Russell-proxy.

Each month, firms are ranked by market cap within the FULL CRSP cross-section
(so ranks correspond to index conventions), then the signal universe is
segmented:  mega = rank <= 500 (S&P 500 proxy),  mid = 501..1000,
small = 1001..3000 (Russell 2000 proxy). Fama-MacBeth and raw long-short
spreads run within each segment.

Tests Lee-et-al (US: effect lives in small firms) against Luo-et-al
(China: survives in large ones) on the repaired panel.

Output: results/size_segments_large.csv
Run:    python pipeline/diag_size_segments.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
import stage08_tests as s8

SEGMENTS = {"mega(SP500proxy)": (1, 500), "mid": (501, 1000),
            "small(R2000proxy)": (1001, 3000)}


def build_panel() -> pd.DataFrame:
    tm = pd.read_parquet(config.DATA_PROCESSED / "techmom_large.parquet")
    tm["month"] = pd.to_datetime(tm["month"])
    r = pd.read_parquet(config.DATA_INTERIM / "returns.parquet")
    r["month"] = pd.to_datetime(r["month"])

    fund = pd.read_parquet(config.DATA_INTERIM / "fundamentals_monthly.parquet")
    fund["month"] = pd.to_datetime(fund["month"])
    r = r.merge(fund, on=["month", "ticker"], how="left")
    mcap = np.exp(r["ln_mcap"])
    r["bm"] = np.where(r["be"] > 0, r["be"] / mcap, np.nan)
    r["roe"] = np.where(r["be"] > 0, r["ni"] / r["be"], np.nan)

    rf = pd.read_parquet(config.DATA_INTERIM / "rf_monthly.parquet")
    rf["month"] = pd.to_datetime(rf["month"])
    fwd = r[["month", "ticker", "ret"]].merge(rf, on="month", how="left")
    fwd["ret"] = fwd["ret"] - fwd["rf"].fillna(0.0)
    fwd = fwd.drop(columns="rf")
    fwd["month"] = fwd["month"] - pd.offsets.MonthEnd(1)
    fwd = fwd.rename(columns={"ret": "fwd_ret"})

    panel = (tm.merge(r, on=["month", "ticker"], how="left")
             .merge(fwd, on=["month", "ticker"], how="inner"))
    num = panel.select_dtypes(include="number").columns
    panel[num] = panel[num].replace([np.inf, -np.inf], np.nan)

    # market-wide size rank from the full CRSP cross-section
    crsp = pd.read_parquet(config.DATA_INTERIM / "returns_crsp.parquet",
                           columns=["month", "ticker", "ln_mcap"])
    crsp["month"] = pd.to_datetime(crsp["month"])
    crsp["mkt_rank"] = (crsp.groupby("month")["ln_mcap"]
                        .rank(ascending=False, method="first"))
    panel = panel.merge(crsp[["month", "ticker", "mkt_rank"]],
                        on=["month", "ticker"], how="left")
    return panel


def main() -> None:
    panel = build_panel()
    rows = []
    for name, (lo, hi) in SEGMENTS.items():
        sub = panel[panel["mkt_rank"].between(lo, hi)]
        n_pm = sub.groupby("month")["ticker"].size().mean()
        fm = s8.fama_macbeth(sub, "techmom_bge")
        r = fm[fm["variable"] == "techmom_bge"].iloc[0]
        ps = s8.portfolio_sort(sub, "techmom_bge", clip_p=None)
        ls = ps[ps["portfolio"] == "Q5-Q1"]
        ew = ls[ls["weighting"] == "EW"].iloc[0]
        vw = ls[ls["weighting"] == "VW"].iloc[0]
        rows.append({"segment": name, "firms/mo": round(n_pm),
                     "fm_coef": round(r["coef"], 4),
                     "fm_t": round(r["t_nw"], 2),
                     "ew_spread": round(ew["mean_fwd_ret"], 4),
                     "ew_t": round(ew["t_nw"], 2),
                     "vw_spread": round(vw["mean_fwd_ret"], 4),
                     "vw_t": round(vw["t_nw"], 2)})
    out = pd.DataFrame(rows)
    out.to_csv(config.RESULTS / "size_segments_large.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
