"""Show pipeline progress from what's on disk — run any time, safe always.

    python check_progress.py
"""

import datetime as dt
from pathlib import Path

import config

EXPECTED_V2_SHARDS = 13


def stamp(p: Path) -> str:
    return dt.datetime.fromtimestamp(p.stat().st_mtime).strftime("%H:%M")


def report(label: str, done: bool, detail: str = "") -> None:
    print(f"  [{'x' if done else ' '}] {label:<38} {detail}")


def main() -> None:
    print("=== raw data ===")
    for name in ["g_patent.tsv.zip", "g_patent_abstract.tsv.zip",
                 "g_assignee_disambiguated.tsv.zip", "g_cpc_current.tsv.zip",
                 "company_tickers.json"]:
        p = config.DATA_RAW / name
        report(name, p.exists(),
               f"{p.stat().st_size/1e6:,.0f} MB" if p.exists() else "missing")

    print("=== interim ===")
    for name in ["patents.parquet", "patent_cpc.parquet", "returns.parquet",
                 "fundamentals_monthly.parquet", "rf_monthly.parquet"]:
        p = config.DATA_INTERIM / name
        report(name, p.exists(), f"updated {stamp(p)}" if p.exists() else "")

    print("=== v1 (bge-small) ===")
    emb = config.DATA_PROCESSED / "embeddings"
    n = len(list(emb.glob("shard_*.npy"))) if emb.exists() else 0
    report("embeddings", (emb / "index.parquet").exists(), f"{n} shards")
    for name in ["corpus_fit_embeddings.npy", "patent_clusters.parquet",
                 "firm_vectors.parquet", "techmom.parquet"]:
        p = config.DATA_PROCESSED / name
        report(name, p.exists(), f"updated {stamp(p)}" if p.exists() else "")

    print("=== v2 (bge-large) ===")
    emb2 = config.DATA_PROCESSED / "embeddings_large"
    n2 = len(list(emb2.glob("shard_*.npy"))) if emb2.exists() else 0
    pct = 100 * n2 // EXPECTED_V2_SHARDS
    newest = max(emb2.glob("shard_*.npy"), key=lambda p: p.stat().st_mtime,
                 default=None) if emb2.exists() else None
    report(f"embeddings_large ({n2}/{EXPECTED_V2_SHARDS} shards, ~{pct}%)",
           n2 >= EXPECTED_V2_SHARDS,
           f"latest shard {stamp(newest)}" if newest else "not started")
    p = config.DATA_PROCESSED / "corpus_fit_embeddings_large.npy"
    report("corpus_fit_embeddings_large.npy", p.exists(),
           f"updated {stamp(p)}" if p.exists() else "")

    print("=== results ===")
    if config.RESULTS.exists():
        for p in sorted(config.RESULTS.glob("*.csv")):
            report(p.name, True, f"updated {stamp(p)}")
    print("\nTip: a shard file appearing every ~20-40 min means embedding is "
          "alive. Task Manager -> Performance -> GPU shows it working live.")


if __name__ == "__main__":
    main()
