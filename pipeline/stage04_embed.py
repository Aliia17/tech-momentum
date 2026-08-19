"""Stage 04 — embed patent abstracts.

Paper: each abstract -> 1024-dim vector from bge-large-zh-v1.5 ([CLS]
pooling, L2-normalized). US analogue: BAAI/bge-*-en-v1.5. This is the
compute-heavy stage: bge-small on CPU does roughly 100-200 abstracts/sec,
so ~1M matched abstracts is a few hours. Work is sharded and resumable —
rerunning skips finished shards.

Outputs
  processed/embeddings/shard_XXXX.npy    float32 [n, dim], L2-normalized
  processed/embeddings/index.parquet     patent_id -> shard, row

Run:  python pipeline/stage04_embed.py [--sample]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config


def main(sample: bool) -> None:
    from sentence_transformers import SentenceTransformer

    src = (config.SAMPLE_DIR if sample else config.DATA_INTERIM) / "patents.parquet"
    out_dir = config.DATA_PROCESSED / ("embeddings_sample" if sample else "embeddings")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(src, columns=["patent_id", "abstract"])
    df = df.reset_index(drop=True)

    # incremental: skip patents already embedded (recorded in index.parquet)
    index_path = out_dir / "index.parquet"
    if index_path.exists():
        old_index = pd.read_parquet(index_path)
        df = df[~df["patent_id"].isin(set(old_index["patent_id"]))]
        df = df.reset_index(drop=True)
        first_shard = int(old_index["shard"].max()) + 1
        print(f"index has {len(old_index):,} embeddings; "
              f"{len(df):,} new abstracts to embed")
    else:
        old_index = None
        first_shard = 0

    n = len(df)
    shard_size = config.EMBED_SHARD_SIZE
    n_shards = (n + shard_size - 1) // shard_size
    print(f"{n:,} abstracts -> {n_shards} shards | model {config.EMBED_MODEL}")

    model = SentenceTransformer(config.EMBED_MODEL)

    index_rows = [] if old_index is None else [old_index]
    for s in range(n_shards):
        lo, hi = s * shard_size, min((s + 1) * shard_size, n)
        shard_id = first_shard + s
        shard_path = out_dir / f"shard_{shard_id:04d}.npy"
        if not shard_path.exists():
            emb = model.encode(
                df["abstract"].iloc[lo:hi].tolist(),
                batch_size=config.EMBED_BATCH_SIZE,
                normalize_embeddings=True,   # L2-normalize, as in the paper
                show_progress_bar=True,
            ).astype(np.float32)
            np.save(shard_path, emb)
        index_rows.append(pd.DataFrame({
            "patent_id": df["patent_id"].iloc[lo:hi],
            "shard": shard_id,
            "row": np.arange(hi - lo),
        }))
        print(f"  shard {s + 1}/{n_shards} done")

    total = sum(len(r) for r in index_rows)
    pd.concat(index_rows, ignore_index=True).to_parquet(index_path, index=False)
    print(f"saved index for {total:,} embeddings in {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="store_true",
                    help="run on synthetic sample data")
    main(ap.parse_args().sample)
