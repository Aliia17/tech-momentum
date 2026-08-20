"""Stage 08 — asset-pricing tests.

Two tests from the paper, run on both TECHMOM measures:

1. Fama-MacBeth: each month, cross-sectional OLS of month t+1 return on
   month-t TECHMOM + controls (ln_mcap, vol, to, rev); time-series means
   of coefficients with Newey-West (3 lags) t-stats. (Paper also controls
   BM, ROE and industry FE — needs Compustat/GICS, noted as a limitation.)

2. Portfolio sorts: TECHMOM orthogonalized on controls each month, stocks
   sorted into quintiles, equal-weighted next-month returns, long-short
   Q5-Q1 with NW t-stat.

Outputs: results/fama_macbeth(_sample).csv, results/portfolio_sorts(_sample).csv

Run:  python pipeline/stage08_tests.py [--sample]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

CONTROLS = ["ln_mcap", "bm", "roe", "vol", "to", "rev"]


def winsorize(s: pd.Series, p: float = 0.01) -> pd.Series:
    return s.clip(s.quantile(p), s.quantile(1 - p))


def nw_tstat(series: pd.Series, lags: int) -> tuple[float, float]:
    """Mean and Newey-West t-stat of a monthly time series."""
    y = series.dropna()
    res = sm.OLS(y, np.ones(len(y))).fit(cov_type="HAC",
                                         cov_kwds={"maxlags": lags})
    return res.params.iloc[0], res.tvalues.iloc[0]


def add_industry_dummies(cs: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame:
    """Append 2-digit-SIC industry dummies when available (paper: industry FE)."""
    if "sic2" in cs.columns and cs["sic2"].notna().sum() > 0:
        dummies = pd.get_dummies(cs["sic2"].fillna("NA"), prefix="ind",
                                 drop_first=True, dtype=float)
        dummies = dummies.loc[:, dummies.sum() >= 5]  # drop tiny industries
        X = pd.concat([X, dummies], axis=1)
    return X


def fama_macbeth(panel: pd.DataFrame, signal: str) -> pd.DataFrame:
    coefs = []
    for month, cs in panel.groupby("month"):
        cols = [signal] + [c for c in CONTROLS if cs[c].notna().sum() > 20]
        cs = cs.dropna(subset=["fwd_ret", signal])
        if len(cs) < config.MIN_FIRMS_PER_MONTH:
            continue
        X = cs[cols].apply(winsorize).fillna(cs[cols].median())
        X = add_industry_dummies(cs, X)
        X = sm.add_constant(X)
        res = sm.OLS(winsorize(cs["fwd_ret"]), X).fit()
        coefs.append(res.params[["const"] + cols].rename(month))
    ts = pd.DataFrame(coefs)
    rows = []
    for col in ts.columns:
        mean, t = nw_tstat(ts[col], config.NEWEY_WEST_LAGS)
        rows.append({"variable": col, "coef": mean, "t_nw": t,
                     "n_months": ts[col].notna().sum()})
    return pd.DataFrame(rows)


def portfolio_sort(panel: pd.DataFrame, signal: str) -> pd.DataFrame:
    """Orthogonalize signal on controls per month, sort into quintiles.

    Paper hygiene applied: cross-sectional winsorization of forward
    returns, micro-cap filter (bottom 20% by mcap dropped monthly, cf.
    the paper's shell-value robustness), and both equal- and value-
    weighted portfolio returns (paper's Table 6 is value-weighted).
    """
    ew: dict[int, dict] = {}
    vw: dict[int, dict] = {}
    for month, cs in panel.groupby("month"):
        cs = cs.dropna(subset=["fwd_ret", signal, "ln_mcap"])
        cs = cs[cs["ln_mcap"] >= cs["ln_mcap"].quantile(0.20)]
        if len(cs) < config.MIN_FIRMS_PER_MONTH:
            continue
        cs = cs.assign(fwd_w=winsorize(cs["fwd_ret"]),
                       mcap=np.exp(cs["ln_mcap"]))
        avail = [c for c in CONTROLS if cs[c].notna().sum() > 20]
        X = cs[avail].apply(winsorize).fillna(cs[avail].median())
        X = add_industry_dummies(cs, X)
        X = sm.add_constant(X)
        resid = sm.OLS(winsorize(cs[signal]), X).fit().resid
        try:
            q = pd.qcut(resid, config.N_PORTFOLIOS, labels=False,
                        duplicates="drop")
        except ValueError:
            continue
        for p in range(config.N_PORTFOLIOS):
            grp = cs.loc[q == p]
            if grp.empty:
                continue
            ew.setdefault(p, {})[month] = grp["fwd_w"].mean()
            vw.setdefault(p, {})[month] = np.average(grp["fwd_w"],
                                                     weights=grp["mcap"])

    rows = []
    for scheme, ports in (("EW", ew), ("VW", vw)):
        for p in sorted(ports):
            mean, t = nw_tstat(pd.Series(ports[p]), config.NEWEY_WEST_LAGS)
            rows.append({"weighting": scheme, "portfolio": f"Q{p + 1}",
                         "mean_fwd_ret": mean, "t_nw": t})
        ls = (pd.Series(ports[max(ports)]) - pd.Series(ports[min(ports)]))
        mean, t = nw_tstat(ls, config.NEWEY_WEST_LAGS)
        rows.append({"weighting": scheme, "portfolio": "Q5-Q1",
                     "mean_fwd_ret": mean, "t_nw": t})
    return pd.DataFrame(rows)


def main(sample: bool, variant: str = "") -> None:
    suffix = "_sample" if sample else (f"_{variant}" if variant else "")
    src_dir = config.SAMPLE_DIR if sample else config.DATA_INTERIM

    techmom = pd.read_parquet(config.DATA_PROCESSED / f"techmom{suffix}.parquet")
    returns = pd.read_parquet(src_dir / "returns.parquet")
    techmom["month"] = pd.to_datetime(techmom["month"])
    returns["month"] = pd.to_datetime(returns["month"])

    # fundamentals (BM, ROE, industry) and risk-free, when stage 03b has run
    fund_path = src_dir / "fundamentals_monthly.parquet"
    if fund_path.exists():
        fund = pd.read_parquet(fund_path)
        fund["month"] = pd.to_datetime(fund["month"])
        returns = returns.merge(fund, on=["month", "ticker"], how="left")
        mcap = np.exp(returns["ln_mcap"])
        returns["bm"] = np.where(returns["be"] > 0, returns["be"] / mcap, np.nan)
        returns["roe"] = np.where(returns["be"] > 0, returns["ni"] / returns["be"],
                                  np.nan)
    rf_path = src_dir / "rf_monthly.parquet"
    rf = None
    if rf_path.exists():
        rf = pd.read_parquet(rf_path)
        rf["month"] = pd.to_datetime(rf["month"])

    # month t signal -> month t+1 return, in excess of the month t+1 risk-free
    fwd = returns[["month", "ticker", "ret"]].copy()
    if rf is not None:
        fwd = fwd.merge(rf, on="month", how="left")
        fwd["ret"] = fwd["ret"] - fwd["rf"].fillna(0.0)
        fwd = fwd.drop(columns="rf")
    fwd["month"] = fwd["month"] - pd.offsets.MonthEnd(1)
    fwd = fwd.rename(columns={"ret": "fwd_ret"})
    panel = (techmom.merge(returns, on=["month", "ticker"], how="left")
             .merge(fwd, on=["month", "ticker"], how="inner"))
    for c in CONTROLS:
        if c not in panel.columns:
            panel[c] = np.nan
    # sanitize: infs (e.g. turnover with zero recorded shares) become NaN,
    # handled downstream by fillna/dropna
    num_cols = panel.select_dtypes(include="number").columns
    panel[num_cols] = panel[num_cols].replace([np.inf, -np.inf], np.nan)
    print(f"test panel: {len(panel):,} firm-months, "
          f"{panel['month'].nunique()} months")

    fm_all, ps_all = [], []
    for signal in ("techmom_bge", "techmom_cls"):
        if signal not in panel.columns or panel[signal].notna().sum() == 0:
            continue
        fm = fama_macbeth(panel, signal)
        fm["signal"] = signal
        fm_all.append(fm)
        ps = portfolio_sort(panel, signal)
        ps["signal"] = signal
        ps_all.append(ps)

    fm = pd.concat(fm_all, ignore_index=True)
    ps = pd.concat(ps_all, ignore_index=True)
    fm.to_csv(config.RESULTS / f"fama_macbeth{suffix}.csv", index=False)
    ps.to_csv(config.RESULTS / f"portfolio_sorts{suffix}.csv", index=False)

    print("\n== Fama-MacBeth ==")
    print(fm.to_string(index=False))
    print("\n== Portfolio sorts (orthogonalized, equal-weighted) ==")
    print(ps.to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--variant", default="", help="e.g. 'large' for bge-large run")
    a = ap.parse_args()
    main(a.sample, a.variant)
