"""Stage 07 — technological links and TECHMOM signal.

Paper eq. (2)-(3): LINK_ijt = cosine(c_it, c_jt) between firm technology
vectors; TECHMOM_it = sum_j LINK_ijt * RET_jt / sum_j LINK_ijt over linked
firms j != i (LINK > 0). Computed each month for both category schemes.

Output: processed/techmom(_sample).parquet
        month, ticker, techmom_bge, techmom_cls

Run:  python pipeline/stage07_techmom.py [--sample]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import normalize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config


def techmom_one_month(vecs: pd.DataFrame, rets: pd.Series) -> pd.Series:
    """vecs: rows (ticker, dim, count) for one month/method.
    rets: month-t return indexed by ticker. Returns TECHMOM by ticker."""
    firms = sorted(set(vecs["ticker"]) & set(rets.index))
    if len(firms) < config.MIN_FIRMS_PER_MONTH:
        return pd.Series(dtype=float)
    vecs = vecs[vecs["ticker"].isin(firms)]
    f_idx = {f: i for i, f in enumerate(firms)}
    d_idx = {d: i for i, d in enumerate(vecs["dim"].unique())}

    m = sparse.csr_matrix(
        (vecs["count"].astype(float),
         (vecs["ticker"].map(f_idx), vecs["dim"].map(d_idx))),
        shape=(len(firms), len(d_idx)),
    )
    m = normalize(m)                       # unit rows -> product is cosine
    link = (m @ m.T).toarray()
    np.fill_diagonal(link, 0.0)            # exclude the focal firm itself

    r = rets.loc[firms].to_numpy()
    denom = link.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        tm = (link @ r) / denom
    return pd.Series(tm, index=firms).dropna()


def main(sample: bool, variant: str = "") -> None:
    suffix = "_sample" if sample else (f"_{variant}" if variant else "")
    src_dir = config.SAMPLE_DIR if sample else config.DATA_INTERIM

    vectors = pd.read_parquet(config.DATA_PROCESSED / f"firm_vectors{suffix}.parquet")
    returns = pd.read_parquet(src_dir / "returns.parquet",
                              columns=["month", "ticker", "ret"])
    returns["month"] = pd.to_datetime(returns["month"])
    vectors["month"] = pd.to_datetime(vectors["month"])

    out_rows = []
    for month, month_vecs in vectors.groupby("month"):
        rets = (returns[returns["month"] == month]
                .set_index("ticker")["ret"].dropna())
        row = {}
        for method, mv in month_vecs.groupby("method"):
            row[method] = techmom_one_month(mv, rets)
        if not row:
            continue
        df = pd.DataFrame({f"techmom_{k}": v for k, v in row.items()})
        df["month"] = month
        out_rows.append(df.rename_axis("ticker").reset_index())

    out = pd.concat(out_rows, ignore_index=True)
    out.to_parquet(config.DATA_PROCESSED / f"techmom{suffix}.parquet", index=False)
    print(f"saved techmom{suffix}.parquet: {len(out):,} firm-months")
    print(out[["techmom_bge", "techmom_cls"]].describe())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--variant", default="", help="e.g. 'large' for bge-large run")
    a = ap.parse_args()
    main(a.sample, a.variant)
