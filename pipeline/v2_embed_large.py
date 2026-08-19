"""V2 — embed everything with bge-large-en-v1.5 (paper-faithful, 1024-dim).

Standalone so it can run in parallel with the v1 (bge-small) pipeline:
waits until v1's GPU work is done (corpus_fit_embeddings.npy exists), then
embeds (a) all linked patent abstracts and (b) the 100k full-corpus
clustering sample, into separate _large outputs. Stages 5-8 get pointed
at these afterwards. Sharded and resumable like stage 04.

Outputs
  processed/embeddings_large/shard_XXXX.npy + index.parquet   [n, 1024]
  processed/corpus_fit_embeddings_large.npy

Run:  python pipeline/v2_embed_large.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

MODEL = "BAAI/bge-large-en-v1.5"
BATCH = 64                      # conservative for 8 GB VRAM at 1024-dim
V1_GPU_DONE_MARKER = config.DATA_PROCESSED / "corpus_fit_embeddings.npy"


def wait_for_gpu() -> None:
    while not V1_GPU_DONE_MARKER.exists():
        print("waiting for v1 GPU work to finish ...", flush=True)
        time.sleep(60)


def embed_patents(model) -> None:
    out_dir = config.DATA_PROCESSED / "embeddings_large"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(config.DATA_INTERIM / "patents.parquet",
                         columns=["patent_id", "abstract"]).reset_index(drop=True)
    n = len(df)
    shard_size = config.EMBED_SHARD_SIZE
    n_shards = (n + shard_size - 1) // shard_size
    print(f"[patents] {n:,} abstracts -> {n_shards} shards | {MODEL}", flush=True)
    index_rows = []
    for s in range(n_shards):
        lo, hi = s * shard_size, min((s + 1) * shard_size, n)
        shard_path = out_dir / f"shard_{s:04d}.npy"
        if not shard_path.exists():
            emb = model.encode(df["abstract"].iloc[lo:hi].tolist(),
                               batch_size=BATCH, normalize_embeddings=True,
                               show_progress_bar=True).astype(np.float32)
            np.save(shard_path, emb)
        index_rows.append(pd.DataFrame({
            "patent_id": df["patent_id"].iloc[lo:hi],
            "shard": s, "row": np.arange(hi - lo)}))
        print(f"[patents] shard {s + 1}/{n_shards} done", flush=True)
    pd.concat(index_rows, ignore_index=True).to_parquet(
        out_dir / "index.parquet", index=False)


def embed_corpus_sample(model) -> None:
    out = config.DATA_PROCESSED / "corpus_fit_embeddings_large.npy"
    if out.exists():
        print("[corpus] already done", flush=True)
        return
    print("[corpus] sampling ids ...", flush=True)
    ids = []
    for chunk in pd.read_csv(
        config.DATA_RAW / "g_patent.tsv.zip", sep="\t", chunksize=1_000_000,
        dtype=str, on_bad_lines="skip",
        usecols=lambda c: c in {"patent_id", "patent_date", "patent_type"},
    ):
        chunk = chunk[(chunk["patent_type"] == "utility")
                      & (chunk["patent_date"] >= "2005-01-01")]
        ids.append(chunk["patent_id"])
    ids = pd.concat(ids, ignore_index=True)
    sampled = set(ids.sample(n=min(len(ids), int(config.CLUSTER_FIT_SAMPLE * 1.5)),
                             random_state=config.SEED))
    texts = []
    for chunk in pd.read_csv(
        config.DATA_RAW / "g_patent_abstract.tsv.zip", sep="\t",
        chunksize=500_000, dtype=str, on_bad_lines="skip",
    ):
        abs_col = [c for c in chunk.columns if "abstract" in c.lower()][0]
        hit = chunk[chunk["patent_id"].isin(sampled)].dropna(subset=[abs_col])
        texts.extend(hit[abs_col].tolist())
        if len(texts) >= config.CLUSTER_FIT_SAMPLE:
            break
    texts = texts[:config.CLUSTER_FIT_SAMPLE]
    print(f"[corpus] embedding {len(texts):,} sampled abstracts", flush=True)
    emb = model.encode(texts, batch_size=BATCH, normalize_embeddings=True,
                       show_progress_bar=True).astype(np.float32)
    np.save(out, emb)


def main() -> None:
    from sentence_transformers import SentenceTransformer
    wait_for_gpu()
    model = SentenceTransformer(MODEL)
    embed_patents(model)
    embed_corpus_sample(model)
    print("V2 EMBEDDINGS COMPLETE", flush=True)


if __name__ == "__main__":
    main()
