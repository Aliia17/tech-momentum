"""Stage 06 — firm-month technology vectors.

Paper eq. (1): at each month-end t, a firm's vector counts its new patents
over the trailing 12 months in each category. Two category schemes:
  bge : K-means cluster of the abstract embedding (stage 05)
  cls : primary CPC subclass (the paper uses IPC subclasses; CPC is the
        scheme PatentsView maintains, same 4-character granularity)

Output: processed/firm_vectors(_sample).parquet
        long format: month, ticker, method, dim, count

Run:  python pipeline/stage06_firm_vectors.py [--sample]
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config


def build(patents: pd.DataFrame, label_col: str, method: str,
          months: pd.DatetimeIndex) -> pd.DataFrame:
    rows = []
    for t in months:
        window = patents[(patents["grant_date"] > t - pd.DateOffset(years=1))
                         & (patents["grant_date"] <= t)]
        if window.empty:
            continue
        counts = (window.groupby(["ticker", label_col], observed=True)
                  .size().rename("count").reset_index())
        counts["month"] = t
        counts["method"] = method
        counts = counts.rename(columns={label_col: "dim"})
        rows.append(counts)
    return pd.concat(rows, ignore_index=True)


def main(sample: bool) -> None:
    suffix = "_sample" if sample else ""
    src_dir = config.SAMPLE_DIR if sample else config.DATA_INTERIM

    patents = pd.read_parquet(src_dir / "patents.parquet",
                              columns=["patent_id", "ticker", "grant_date"])
    patents["grant_date"] = pd.to_datetime(patents["grant_date"])
    clusters = pd.read_parquet(
        config.DATA_PROCESSED / f"patent_clusters{suffix}.parquet")
    cpc = pd.read_parquet(src_dir / "patent_cpc.parquet")

    months = pd.date_range(config.SIGNAL_START, config.SIGNAL_END, freq="ME")

    bge = build(patents.merge(clusters, on="patent_id"), "cluster", "bge", months)
    cls = build(patents.merge(cpc, on="patent_id"), "cpc_subclass", "cls", months)

    out = pd.concat([bge, cls], ignore_index=True)
    out["dim"] = out["dim"].astype(str)
    out.to_parquet(config.DATA_PROCESSED / f"firm_vectors{suffix}.parquet",
                   index=False)
    print(f"saved firm_vectors{suffix}.parquet: {len(out):,} rows | "
          f"months {out['month'].nunique()} | firms {out['ticker'].nunique():,}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="store_true")
    main(ap.parse_args().sample)
