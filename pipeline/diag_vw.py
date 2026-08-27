"""Diagnostic — hunt the corrupted value weights.

Symptom: Yahoo-based value-weighted quintiles average -0.51%/month
2010-2024, which is impossible for a cap-weighted book of large US patent
holders. Hypothesis: a few firms carry garbage shares-outstanding from
Yahoo fast_info, creating fake mega-caps whose returns dominate the VW
average.

Checks:
  1. Panel-wide VW monthly mean return (should be ~+1%/mo if weights real)
  2. Top-20 firms by average VW weight share, with implied market cap
     (fakes reveal themselves: implied mcap wildly off known values)
  3. Same table after dropping suspects, VW mean recomputed

Run:  python pipeline/diag_vw.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config


def main() -> None:
    r = pd.read_parquet(config.DATA_INTERIM / "returns.parquet")
    r["month"] = pd.to_datetime(r["month"])
    r = r[(r["month"] >= "2010-01-01") & (r["month"] <= "2024-12-31")]
    r = r.dropna(subset=["ret", "ln_mcap"])
    r = r[np.isfinite(r["ln_mcap"])]
    r["mcap"] = np.exp(r["ln_mcap"])

    def vw_mean(df: pd.DataFrame) -> float:
        m = df.groupby("month").apply(
            lambda g: np.average(g["ret"].clip(-0.9, 3.0), weights=g["mcap"]),
            include_groups=False)
        return float(m.mean())

    print(f"panel VW monthly mean return : {vw_mean(r):+.4f}")
    ew = r.groupby("month")["ret"].mean().mean()
    print(f"panel EW monthly mean return : {ew:+.4f}")

    # average weight share per ticker
    r["w"] = r["mcap"] / r.groupby("month")["mcap"].transform("sum")
    top = (r.groupby("ticker")
           .agg(avg_w=("w", "mean"), med_mcap_bn=("mcap", "median"),
               mean_ret=("ret", "mean"), months=("w", "size"))
           .sort_values("avg_w", ascending=False).head(20))
    top["med_mcap_bn"] = (top["med_mcap_bn"] / 1e9).round(1)
    top["avg_w"] = (100 * top["avg_w"]).round(2)
    top["mean_ret"] = (100 * top["mean_ret"]).round(2)
    print("\ntop 20 by average VW weight (avg_w in %, med mcap in $bn, mean monthly ret in %):")
    print(top.to_string())

    # recompute VW without any firm whose median implied mcap > $4tn (impossible)
    fakes = top[top["med_mcap_bn"] > 4000].index.tolist()
    print(f"\nfirms with median implied mcap > $4tn (impossible): {fakes}")
    if fakes:
        print(f"panel VW mean after dropping them: {vw_mean(r[~r['ticker'].isin(fakes)]):+.4f}")


if __name__ == "__main__":
    main()
