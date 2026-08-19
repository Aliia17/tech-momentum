# Embed patent abstracts with bge-large-en-v1.5 on a free Colab GPU.
#
# How to use (10 minutes of clicking, ~2-4 h of GPU time):
#   1. Go to https://colab.research.google.com -> New notebook
#   2. Runtime -> Change runtime type -> T4 GPU
#   3. Upload data/interim/patents.parquet to the Colab session
#      (left sidebar -> Files -> Upload), or mount Google Drive and
#      put the file there.
#   4. Paste this whole file into one cell and run it.
#   5. When it finishes, download the "embeddings_large" folder
#      (it zips itself at the end) and drop its contents into
#      tech-momentum/data/processed/embeddings_large/ on your laptop.
#   6. Tell Claude to rerun stages 5-8 against embeddings_large.

# %pip install -q sentence-transformers pyarrow

from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

MODEL = "BAAI/bge-large-en-v1.5"     # 1024-dim, the paper-faithful analogue
SHARD_SIZE = 100_000
BATCH_SIZE = 256                      # T4 GPU handles this comfortably
SRC = Path("patents.parquet")         # uploaded file
OUT = Path("embeddings_large")
OUT.mkdir(exist_ok=True)

df = pd.read_parquet(SRC, columns=["patent_id", "abstract"]).reset_index(drop=True)
n = len(df)
n_shards = (n + SHARD_SIZE - 1) // SHARD_SIZE
print(f"{n:,} abstracts -> {n_shards} shards | {MODEL}")

model = SentenceTransformer(MODEL, device="cuda")

index_rows = []
for s in range(n_shards):
    lo, hi = s * SHARD_SIZE, min((s + 1) * SHARD_SIZE, n)
    shard_path = OUT / f"shard_{s:04d}.npy"
    if not shard_path.exists():          # resumable if Colab disconnects
        emb = model.encode(
            df["abstract"].iloc[lo:hi].tolist(),
            batch_size=BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=True,
        ).astype(np.float32)
        np.save(shard_path, emb)
    index_rows.append(pd.DataFrame({
        "patent_id": df["patent_id"].iloc[lo:hi],
        "shard": s,
        "row": np.arange(hi - lo),
    }))
    print(f"shard {s + 1}/{n_shards} done")

pd.concat(index_rows, ignore_index=True).to_parquet(OUT / "index.parquet",
                                                    index=False)
print("zipping for download ...")
import shutil
shutil.make_archive("embeddings_large", "zip", OUT)
print("done -> download embeddings_large.zip from the Files sidebar")
