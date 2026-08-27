"""V3 — K-free TECHMOM: continuous embedding similarity, no clustering.

Reviewer point 3: the SSE curve is a scale-free power law (log-log slope
flat across K = 50..2000), so no cluster count is data-privileged and the
K=500 discretization is a convention. This variant removes the choice
entirely: a firm's technology vector at month t is the MEAN of its
trailing-12-month patents' bge-large embeddings (L2-normalized), and
LINK_ij is the cosine between firm mean vectors. TECHMOM as usual.
If results match the K-variants, conclusions do not rest on K.

Output: processed/techmom_cont.parquet (column techmom_bge = continuous)
Run:    python pipeline/v3_continuous_link.py     (then stage08 --variant cont)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

EMB_DIR = config.DATA_PROCESSED / "embeddings_large"
DIM = 1024


def main() -> None:
    patents = pd.read_parquet(config.DATA_INTERIM / "patents.parquet",
                              columns=["patent_id", "ticker", "grant_date"])
    patents["grant_date"] = pd.to_datetime(patents["grant_date"])
    patents["gmonth"] = patents["grant_date"] + pd.offsets.MonthEnd(0)

    index = (pd.read_parquet(EMB_DIR / "index.parquet")
             .sort_values(["shard", "row"]).reset_index(drop=True))
    meta = index.merge(patents, on="patent_id", how="inner")

    tickers = sorted(meta["ticker"].unique())
    months = pd.date_range(config.SIGNAL_START, config.SIGNAL_END, freq="ME")
    # include the 12 months before the first signal month (rolling window)
    all_months = pd.date_range(months[0] - pd.DateOffset(years=1),
                               months[-1], freq="ME")
    t_idx = {t: i for i, t in enumerate(tickers)}
    m_idx = {m: i for i, m in enumerate(all_months)}

    # one pass over the shards: accumulate per (ticker, grant-month) embedding
    # sums and counts
    print(f"accumulating {len(meta):,} embeddings into "
          f"{len(tickers)} x {len(all_months)} monthly sums ...")
    sums = np.zeros((len(tickers), len(all_months), DIM), dtype=np.float32)
    counts = np.zeros((len(tickers), len(all_months)), dtype=np.int32)
    meta = meta[meta["gmonth"].isin(set(all_months))]
    for shard_id, grp in meta.groupby("shard"):
        X = np.load(EMB_DIR / f"shard_{shard_id:04d}.npy")
        ti = grp["ticker"].map(t_idx).to_numpy()
        mi = grp["gmonth"].map(m_idx).to_numpy()
        rows = grp["row"].to_numpy()
        np.add.at(sums, (ti, mi), X[rows])
        np.add.at(counts, (ti, mi), 1)
        print(f"  shard {shard_id} done")

    # rolling 12-month window per month-end -> firm mean vector -> cosine LINK
    print("computing monthly continuous TECHMOM ...")
    returns = pd.read_parquet(config.DATA_INTERIM / "returns.parquet",
                              columns=["month", "ticker", "ret"])
    returns["month"] = pd.to_datetime(returns["month"])

    out = []
    for t in months:
        j = m_idx[t]
        lo = j - 11
        win_sum = sums[:, lo:j + 1].sum(axis=1)
        win_n = counts[:, lo:j + 1].sum(axis=1)
        active = np.where(win_n > 0)[0]
        rets = (returns[returns["month"] == t]
                .set_index("ticker")["ret"].dropna())
        firms = [tickers[i] for i in active if tickers[i] in rets.index]
        if len(firms) < config.MIN_FIRMS_PER_MONTH:
            continue
        rows_i = [t_idx[f] for f in firms]
        V = win_sum[rows_i] / win_n[rows_i, None]
        V /= np.linalg.norm(V, axis=1, keepdims=True)
        link = V @ V.T
        np.fill_diagonal(link, 0.0)
        link = np.clip(link, 0.0, None)      # negative cosines: not links
        r = rets.loc[firms].to_numpy()
        denom = link.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            tm = (link @ r) / denom
        out.append(pd.DataFrame({"month": t, "ticker": firms,
                                 "techmom_bge": tm}))

    res = pd.concat(out, ignore_index=True).dropna()
    res.to_parquet(config.DATA_PROCESSED / "techmom_cont.parquet", index=False)
    print(f"saved techmom_cont.parquet: {len(res):,} firm-months, "
          f"{res['month'].nunique()} months")


if __name__ == "__main__":
    main()
