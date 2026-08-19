"""Stage 04b — embed a random sample of the FULL patent corpus for clustering.

The paper fits K-means on 100k patents sampled from its entire corpus, not
just firm-linked ones, so the 500 technology themes represent the whole
technology space. This stage reproduces that: sample CLUSTER_FIT_SAMPLE
random utility patents (2005+) from the complete g_patent_abstract table
(~8M patents, linked or not), embed them, and save the matrix. Stage 05
fits cluster centers on this file when it exists, then assigns the
firm-linked patents to those centers.

Output: processed/corpus_fit_embeddings.npy

Run:  python pipeline/stage04b_corpus_sample.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

CORPUS_START = "2005-01-01"   # sample modern-vocabulary patents


def main() -> None:
    from sentence_transformers import SentenceTransformer

    out = config.DATA_PROCESSED / "corpus_fit_embeddings.npy"
    if out.exists():
        print(f"{out.name} already exists, skipping")
        return

    print("collecting candidate patent ids from g_patent ...")
    ids = []
    for chunk in pd.read_csv(
        config.DATA_RAW / "g_patent.tsv.zip",
        sep="\t", chunksize=1_000_000, dtype=str,
        usecols=lambda c: c in {"patent_id", "patent_date", "patent_type"},
        on_bad_lines="skip",
    ):
        chunk = chunk[(chunk["patent_type"] == "utility")
                      & (chunk["patent_date"] >= CORPUS_START)]
        ids.append(chunk["patent_id"])
    ids = pd.concat(ids, ignore_index=True)
    print(f"  candidates: {len(ids):,}")

    rng = np.random.default_rng(config.SEED)
    # oversample: some sampled ids will lack abstracts
    n_draw = min(len(ids), int(config.CLUSTER_FIT_SAMPLE * 1.5))
    sampled = set(ids.sample(n=n_draw, random_state=config.SEED))

    print("collecting sampled abstracts from g_patent_abstract ...")
    texts = []
    for chunk in pd.read_csv(
        config.DATA_RAW / "g_patent_abstract.tsv.zip",
        sep="\t", chunksize=500_000, dtype=str, on_bad_lines="skip",
    ):
        abs_col = [c for c in chunk.columns if "abstract" in c.lower()][0]
        hit = chunk[chunk["patent_id"].isin(sampled)].dropna(subset=[abs_col])
        texts.extend(hit[abs_col].tolist())
        if len(texts) >= config.CLUSTER_FIT_SAMPLE:
            break
    texts = texts[:config.CLUSTER_FIT_SAMPLE]
    print(f"  sampled abstracts: {len(texts):,}")

    model = SentenceTransformer(config.EMBED_MODEL)
    emb = model.encode(texts, batch_size=config.EMBED_BATCH_SIZE,
                       normalize_embeddings=True,
                       show_progress_bar=True).astype(np.float32)
    np.save(out, emb)
    print(f"saved {out.name}: {emb.shape}")


if __name__ == "__main__":
    main()
