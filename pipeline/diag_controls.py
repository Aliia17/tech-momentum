"""Diagnostic — does control-variable quality drive the TECHMOM result?

Reruns the full-sample Fama-MacBeth under three control sets:
  full      ln_mcap, bm, roe, vol, to, rev   (headline spec)
  price     ln_mcap, vol, to, rev            (drop the EDGAR proxies bm/roe)
  minimal   ln_mcap, rev                     (drop turnover too)

If the TECHMOM coefficient/t-stat barely moves, the fundamental proxies are
not what separates our results from the paper's.

Run:  python pipeline/diag_controls.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
import stage08_tests as s8
from v1_subsample_fm import build_panel

SETS = {
    "full":    ["ln_mcap", "bm", "roe", "vol", "to", "rev"],
    "price":   ["ln_mcap", "vol", "to", "rev"],
    "minimal": ["ln_mcap", "rev"],
}


def main() -> None:
    panel = build_panel()
    rows = []
    for label, controls in SETS.items():
        s8.CONTROLS = controls
        for signal in ("techmom_bge", "techmom_cls"):
            fm = s8.fama_macbeth(panel, signal)
            r = fm[fm["variable"] == signal].iloc[0]
            rows.append({"controls": label, "signal": signal,
                         "coef": round(r["coef"], 4), "t_nw": round(r["t_nw"], 2)})
    out = pd.DataFrame(rows)
    out.to_csv(config.RESULTS / "fm_control_sensitivity.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
