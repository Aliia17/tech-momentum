"""Diagnostic — early vs late subperiods for the bge-large signal.

Did the US premium exist in the early years (2010-2014: after Lee et al.'s
sample ends, before their 2019 publication) and die later? Fama-MacBeth and
raw long-short spreads per subperiod.

Output: results/subsample_large.csv
Run:    python pipeline/diag_subsample_large.py [split-date, default 2015-01-01]
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
import stage08_tests as s8
from diag_size_segments import build_panel

SPLIT = sys.argv[1] if len(sys.argv) > 1 else "2015-01-01"


def main() -> None:
    panel = build_panel()
    rows = []
    for label, sub in ((f"early 2010..{SPLIT[:4]}", panel[panel["month"] < SPLIT]),
                       (f"late {SPLIT[:4]}..2024", panel[panel["month"] >= SPLIT])):
        fm = s8.fama_macbeth(sub, "techmom_bge")
        r = fm[fm["variable"] == "techmom_bge"].iloc[0]
        ps = s8.portfolio_sort(sub, "techmom_bge", clip_p=None)
        ls = ps[ps["portfolio"] == "Q5-Q1"]
        ew = ls[ls["weighting"] == "EW"].iloc[0]
        vw = ls[ls["weighting"] == "VW"].iloc[0]
        rows.append({"period": label, "months": int(r["n_months"]),
                     "fm_coef": round(r["coef"], 4),
                     "fm_t": round(r["t_nw"], 2),
                     "ew_spread": round(ew["mean_fwd_ret"], 4),
                     "ew_t": round(ew["t_nw"], 2),
                     "vw_spread": round(vw["mean_fwd_ret"], 4),
                     "vw_t": round(vw["t_nw"], 2)})
    out = pd.DataFrame(rows)
    out.to_csv(config.RESULTS / "subsample_large.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
