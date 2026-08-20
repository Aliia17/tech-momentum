"""Stage 05 — cluster embeddings into technology categories.

Paper: K-means with K=500 fit on a random sample of 100,000 patent vectors
(K chosen at the SSE elbow, comparable to the 651 IPC subclasses); all
remaining patents assigned to the nearest cluster center.

Outputs
  processed/patent_clusters(.sample).parquet   patent_id, cluster
  results/kmeans_sse(.sample).csv              SSE by K (elbow diagnostic)

Run:  python pipeline/stage05_cluster.py [--sample]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config


def load_embeddings(emb_dir: Path) -> tuple[np.ndarray, pd.DataFrame]:
    # index must align row-for-row with the vstacked shards
    index = (pd.read_parquet(emb_dir / "index.parquet")
             .sort_values(["shard", "row"]).reset_index(drop=True))
    shards = [np.load(emb_dir / f"shard_{s:04d}.npy")
              for s in sorted(index["shard"].unique())]
    return np.vstack(shards), index


def main(sample: bool, variant: str = "", emb: str = "", k: int = 0) -> None:
    """variant tags the OUTPUT files; emb picks the embedding source dir
    (defaults to variant); k overrides config.N_CLUSTERS."""
    suffix = "_sample" if sample else (f"_{variant}" if variant else "")
    emb_suffix = "_sample" if sample else (f"_{emb}" if emb else suffix)
    emb_dir = config.DATA_PROCESSED / f"embeddings{emb_suffix}"
    X, index = load_embeddings(emb_dir)
    print(f"embeddings: {X.shape}")

    # Fit sample: prefer the full-corpus random sample (stage 04b) so the
    # 500 technology themes represent the whole patent space, as in the
    # paper; fall back to sampling the firm-linked embeddings.
    corpus_path = config.DATA_PROCESSED / f"corpus_fit_embeddings{emb_suffix}.npy"
    rng = np.random.default_rng(config.SEED)
    if not sample and corpus_path.exists():
        fit_X = np.load(corpus_path)
        print(f"fitting on full-corpus sample: {fit_X.shape}")
    else:
        n_fit = min(config.CLUSTER_FIT_SAMPLE, len(X))
        fit_X = X[rng.choice(len(X), size=n_fit, replace=False)]
        print(f"fitting on sample of linked embeddings: {fit_X.shape}")
    n_fit = len(fit_X)
    k = min(k or config.N_CLUSTERS, max(2, n_fit // 20))

    # elbow diagnostic around the chosen K (cheap, coarse grid)
    sse_rows = []
    for k_try in sorted({max(2, k // 5), max(3, k // 2), k, k * 2}):
        if k_try >= n_fit:
            continue
        km_try = MiniBatchKMeans(n_clusters=k_try, random_state=config.SEED,
                                 n_init=3, batch_size=4096).fit(fit_X)
        sse_rows.append({"k": k_try, "sse": km_try.inertia_})
    pd.DataFrame(sse_rows).to_csv(
        config.RESULTS / f"kmeans_sse{suffix}.csv", index=False)

    print(f"fitting K-means: K={k} on {n_fit:,} sampled vectors")
    km = MiniBatchKMeans(n_clusters=k, random_state=config.SEED,
                         n_init=10, batch_size=4096).fit(fit_X)

    # assign every patent to its nearest center, in chunks to bound memory
    labels = np.empty(len(X), dtype=np.int32)
    step = 200_000
    for lo in range(0, len(X), step):
        labels[lo:lo + step] = km.predict(X[lo:lo + step])

    out = index[["patent_id"]].copy()
    out["cluster"] = labels
    out.to_parquet(config.DATA_PROCESSED / f"patent_clusters{suffix}.parquet",
                   index=False)
    occupied = pd.Series(labels).nunique()
    print(f"saved clusters for {len(out):,} patents ({occupied}/{k} clusters occupied)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--variant", default="", help="output tag, e.g. 'large'")
    ap.add_argument("--emb", default="", help="embedding source tag (default: variant)")
    ap.add_argument("--k", type=int, default=0, help="override N_CLUSTERS")
    a = ap.parse_args()
    main(a.sample, a.variant, a.emb, a.k)
