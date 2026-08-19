"""Subsample Fama-MacBeth — alpha-decay check.

Splits the test window into early (2010-2016) and late (2017-2024) halves
and reruns the Fama-MacBeth regressions per half. If technological momentum
was arbitraged away after Lee et al. (2019) publicized it, the early-half
coefficients should dominate.

Output: results/fama_macbeth_subsamples.csv
Run:    python pipeline/v1_subsample_fm.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from stage08_tests import CONTROLS, fama_macbeth

SPLIT = "2017-01-01"


def build_panel() -> pd.DataFrame:
    techmom = pd.read_parquet(config.DATA_PROCESSED / "techmom.parquet")
    returns = pd.read_parquet(config.DATA_INTERIM / "returns.parquet")
    techmom["month"] = pd.to_datetime(techmom["month"])
    returns["month"] = pd.to_datetime(returns["month"])

    fund = pd.read_parquet(config.DATA_INTERIM / "fundamentals_monthly.parquet")
    fund["month"] = pd.to_datetime(fund["month"])
    returns = returns.merge(fund, on=["month", "ticker"], how="left")
    mcap = np.exp(returns["ln_mcap"])
    returns["bm"] = np.where(returns["be"] > 0, returns["be"] / mcap, np.nan)
    returns["roe"] = np.where(returns["be"] > 0, returns["ni"] / returns["be"], np.nan)

    rf = pd.read_parquet(config.DATA_INTERIM / "rf_monthly.parquet")
    rf["month"] = pd.to_datetime(rf["month"])

    fwd = returns[["month", "ticker", "ret"]].merge(rf, on="month", how="left")
    fwd["ret"] = fwd["ret"] - fwd["rf"].fillna(0.0)
    fwd = fwd.drop(columns="rf")
    fwd["month"] = fwd["month"] - pd.offsets.MonthEnd(1)
    fwd = fwd.rename(columns={"ret": "fwd_ret"})

    panel = (techmom.merge(returns, on=["month", "ticker"], how="left")
             .merge(fwd, on=["month", "ticker"], how="inner"))
    num = panel.select_dtypes(include="number").columns
    panel[num] = panel[num].replace([np.inf, -np.inf], np.nan)
    return panel


def main() -> None:
    panel = build_panel()
    out = []
    for label, sub in (
        (f"early (<{SPLIT[:4]})", panel[panel["month"] < SPLIT]),
        (f"late (>={SPLIT[:4]})", panel[panel["month"] >= SPLIT]),
    ):
        for signal in ("techmom_bge", "techmom_cls"):
            fm = fama_macbeth(sub, signal)
            fm = fm[fm["variable"] == signal].assign(sample=label)
            out.append(fm)
    res = pd.concat(out, ignore_index=True)[
        ["sample", "variable", "coef", "t_nw", "n_months"]]
    res.to_csv(config.RESULTS / "fama_macbeth_subsamples.csv", index=False)
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
