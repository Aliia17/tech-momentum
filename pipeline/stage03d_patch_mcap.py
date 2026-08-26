"""Stage 03d — repair market caps in the Yahoo returns panel.

Root cause (found via diag_vw.py, prompted by external review): market cap
was computed as ADJUSTED price x CURRENT shares outstanding. For serial
reverse-splitters, Yahoo's adjustment inflates past prices by the cumulative
split factor, manufacturing fake mega-caps (implied $316tn for a nano-cap)
that dominated every value-weighted portfolio.

Repairs, in order:
  1. drop non-finite returns (a literal +inf survived in the panel)
  2. where CRSP has the (month, ticker): replace ln_mcap and turnover with
     CRSP-based values (true historical shares x true price)
  3. residual Yahoo-only rows: null ln_mcap when implied mcap > $5tn or
     when the firm's own mcap path moves >100x month-to-month (split ghosts)

Writes returns.parquet in place (backup: returns_prepatch.parquet).
Run:  python pipeline/stage03d_patch_mcap.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

MCAP_CEILING = 5e12          # no company is worth > $5tn in-sample

# Non-CRSP tickers whose market caps we trust anyway: established foreign
# issuers with verified US listings (the ADR class the alias table added).
# Anything else outside CRSP gets returns only, no value weight.
ADR_WHITELIST = {
    "TM", "TSM", "SONY", "ASML", "SAP", "NVS", "AZN", "GSK", "SNY",
    "ERIC", "PHG", "STM", "NOK", "NXPI", "HMC", "SHEL", "BP", "TTE",
    "UL", "BTI", "HSBC", "MUFG", "BABA", "PDD", "JD", "NTES", "BIDU",
    "INFY", "WIT", "VALE", "RIO", "BHP", "ARM", "IGT", "BB",
}


def main() -> None:
    path = config.DATA_INTERIM / "returns.parquet"
    backup = config.DATA_INTERIM / "returns_prepatch.parquet"
    # idempotent: always repair from the original panel
    r = pd.read_parquet(backup if backup.exists() else path)
    r["month"] = pd.to_datetime(r["month"])
    n0 = len(r)

    # 1. finite returns only
    r = r[np.isfinite(r["ret"])]
    print(f"dropped {n0 - len(r)} non-finite return rows")

    # 2. CRSP-first: wherever CRSP covers the (month, ticker), its return,
    # market cap and turnover replace Yahoo's (dividend-inclusive returns,
    # true historical shares, audited prices). Yahoo remains only for what
    # CRSP cannot cover: foreign ADRs and post-2023 months.
    crsp = pd.read_parquet(config.DATA_INTERIM / "returns_crsp.parquet",
                           columns=["month", "ticker", "ret", "ln_mcap", "to"])
    crsp["month"] = pd.to_datetime(crsp["month"])
    crsp = crsp.rename(columns={c: f"{c}_crsp" for c in ("ret", "ln_mcap", "to")})
    r = r.merge(crsp, on=["month", "ticker"], how="left")
    n_crsp = r["ln_mcap_crsp"].notna().sum()
    for c in ("ret", "ln_mcap", "to"):
        r[c] = r[f"{c}_crsp"].fillna(r[c])
        r = r.drop(columns=f"{c}_crsp")
    r["rev"] = r["ret"]
    print(f"CRSP override (ret/mcap/to): {n_crsp:,} of {len(r):,} firm-months")

    # 2b. quarantine corrupt tickers: never covered by CRSP (so neither US
    # common stock nor recent) AND showing the reverse-split corruption
    # signature (absurd monthly returns) or untradeable size. Legit foreign
    # ADRs (TM, TSM, ...) have no CRSP coverage but sane returns and size.
    cov = r.groupby("ticker")["ln_mcap"].transform("size")  # placeholder
    crsp_cov = r.assign(has=r["ticker"].isin(set(crsp["ticker"]))) \
                .groupby("ticker")["has"].transform("any")
    stats = r.groupby("ticker").agg(max_ret=("ret", "max"),
                                    mean_ret=("ret", "mean"),
                                    med_mcap=("ln_mcap",
                                              lambda s: np.exp(s.median())))
    corrupt = stats[(stats["max_ret"] > 2.0) | (stats["med_mcap"] < 5e7)
                    | (stats["mean_ret"] < -0.03)].index
    quarantine = r["ticker"].isin(corrupt) & ~crsp_cov
    print(f"quarantined {quarantine.sum():,} firm-months from "
          f"{r.loc[quarantine, 'ticker'].nunique()} corrupt non-CRSP tickers")
    r = r[~quarantine]

    # 3. sanity-null the residual Yahoo-only market caps
    mcap = np.exp(r["ln_mcap"])
    bad_ceiling = mcap > MCAP_CEILING
    jump = (r.sort_values(["ticker", "month"]).groupby("ticker")["ln_mcap"]
            .diff().abs() > np.log(10))
    jump = jump.reindex(r.index, fill_value=False)
    # Yahoo-only months of CRSP-covered tickers: cap at 20x the ticker's
    # own CRSP-era median (catches reverse-split ghosts in 2024 months)
    med = r.groupby("ticker")["ln_mcap"].transform("median")
    off_median = (r["ln_mcap"] - med) > np.log(20)
    # value weights only for verifiable caps: CRSP-covered or whitelisted ADR
    crsp_cov2 = r.assign(has=r["ticker"].isin(set(crsp["ticker"]))) \
                 .groupby("ticker")["has"].transform("any")
    unverified = ~crsp_cov2 & ~r["ticker"].isin(ADR_WHITELIST)
    bad = bad_ceiling | jump | off_median | unverified
    r.loc[bad, "ln_mcap"] = np.nan
    print(f"nulled {int(bad.sum()):,} implausible ln_mcap values "
          f"({int(bad_ceiling.sum())} ceiling, {int(jump.sum())} jumps, "
          f"{int(off_median.sum())} off-median)")
    # returns sanity for rows CRSP could not verify: clip lottery artifacts
    wild = ~np.isfinite(r["ret"]) | (r["ret"] > 3.0) | (r["ret"] < -0.95)
    print(f"dropped {int(wild.sum()):,} implausible return rows (|ret| wild)")
    r = r[~wild]

    backup = config.DATA_INTERIM / "returns_prepatch.parquet"
    if not backup.exists():
        pd.read_parquet(path).to_parquet(backup, index=False)
    r.to_parquet(path, index=False)

    # verify: panel-wide VW mean must now be plausible
    chk = r.dropna(subset=["ret", "ln_mcap"])
    chk = chk[(chk["month"] >= "2010-01-01")]
    vw = (chk.assign(mcap=np.exp(chk["ln_mcap"]))
          .groupby("month")
          .apply(lambda g: np.average(g["ret"], weights=g["mcap"]),
                 include_groups=False).mean())
    print(f"panel VW monthly mean after repair: {vw:+.4f}  (sanity: ~+0.01)")


if __name__ == "__main__":
    main()
