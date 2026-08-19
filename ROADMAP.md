# Pipeline Roadmap — who runs what, in which order, and why

This is the operator's manual. It assumes nothing beyond: Python 3.12, the
packages in `requirements.txt`, disk space (~15 GB), and optionally an NVIDIA
GPU (10-50x faster embeddings; CPU works).

## The picture

```
 STAGE 1          STAGE 2              STAGE 3 / 3b            STAGE 4 / 4b
 download   ->    link patents    ->   returns +          ->   embed abstracts
 raw data         to tickers          fundamentals            (GPU if available)
    |                 |                    |                       |
 data/raw/        patents.parquet     returns.parquet         embeddings/*.npy
 (5 files)        patent_cpc.parquet  fundamentals_monthly    corpus_fit_embeddings
                                      rf_monthly
                                           \                       |
                                            \                 STAGE 5
                                             \                K-means, 500 themes
                                              \                    |
                                               \              patent_clusters
                                                \                  |
                                                 ->  STAGE 6: firm vectors (12m rolling)
                                                          |
                                                     STAGE 7: LINK + TECHMOM signal
                                                          |
                                                     STAGE 8: Fama-MacBeth + sorts
                                                          |
                                                     results/*.csv
```

One command runs everything in order: `python run_pipeline.py`
(or `--sample` for the 5-minute synthetic smoke test — run this FIRST on any
new machine; if it fails, fix the environment before burning hours).

## Stage-by-stage: needs -> makes -> time

| # | Script | Needs | Makes | Time |
|---|--------|-------|-------|------|
| 1 | `stage01_download.py` | internet | `data/raw/`: g_patent, g_patent_abstract, g_assignee_disambiguated, g_cpc_current (.tsv.zip, Zenodo-frozen 12/2024) + company_tickers.json (SEC) | ~30-60 min (2.7 GB) |
| 2 | `stage02_link_firms.py` | stage 1 | `data/interim/patents.parquet` (patent_id, grant_date, abstract, ticker), `patent_cpc.parquet`, `match_report.txt` | ~15 min |
| 3 | `stage03_returns.py` | stage 2 | `data/interim/returns.parquet` (month, ticker, ret, ln_mcap, vol, to, rev) | ~40 min (Yahoo) |
| 3b | `stage03b_fundamentals.py` | stage 2 | `data/interim/fundamentals_monthly.parquet` (be, ni, sic2, point-in-time), `rf_monthly.parquet` | ~30 min (SEC EDGAR) |
| 4 | `stage04_embed.py` | stage 2 | `data/processed/embeddings/shard_*.npy` + `index.parquet` | GPU ~1 h / CPU ~10 h |
| 4b | `stage04b_corpus_sample.py` | stage 1 | `data/processed/corpus_fit_embeddings.npy` (100k random full-corpus patents — the K-means fit sample) | GPU ~10 min / CPU ~1 h |
| 5 | `stage05_cluster.py` | 4, 4b | `data/processed/patent_clusters.parquet` (patent_id -> cluster 0..499), `results/kmeans_sse.csv` | ~10 min |
| 6 | `stage06_firm_vectors.py` | 2, 5 | `data/processed/firm_vectors.parquet` (month, ticker, method, dim, count) | ~10 min |
| 7 | `stage07_techmom.py` | 3, 6 | `data/processed/techmom.parquet` (month, ticker, techmom_bge, techmom_cls) | ~5 min |
| 8 | `stage08_tests.py` | 3, 3b, 7 | `results/fama_macbeth.csv`, `results/portfolio_sorts.csv` (printed to console too) | ~5 min |

Extras (not in the default chain):
- `pipeline/make_sample_data.py` — synthetic world for `--sample` runs
- `pipeline/diag_unmatched.py` — ranks unmatched assignees (feeds the alias table)
- `pipeline/v1_subsample_fm.py` — early/late alpha-decay split
- `pipeline/v2_embed_large.py` — bge-large (1024-dim) re-embed for the v2 comparison
- `colab_embed_large.py` — same as v2 but on a free Colab GPU

Restart rules: every stage is idempotent — rerunning skips or overwrites
cleanly. Embedding (4/4b/v2) checkpoints per 100k shard: a killed run resumes
where it left off. Stage 2's output feeds everything, so after changing the
alias table rerun 2 -> 3 -> 3b -> 4 (incremental) -> 5 -> 6 -> 7 -> 8.

## Where a human inserts the API key (live data after 12/2024)

The patent snapshot is DOI-frozen at 12/31/2024 (same endpoint as the paper).
For fresh grants a free USPTO Open Data Portal key is needed (data.uspto.gov;
registration requires identity verification — a one-time human step).

Insert the key ONE of two ways (both read by `config.ODP_API_KEY`):
1. environment variable `USPTO_ODP_API_KEY`, or
2. paste it into a file `tech-momentum/.odp_api_key` (gitignored; never
   committed).

Then the monthly update cycle is: fetch grants since the last snapshot
(grant date, abstract, assignee, CPC) -> append to raw -> rerun stages
2, 4 (incremental — only new patents get embedded), 6, 7 with the NEW months
-> stage 3/3b refresh -> stage 8 if re-testing. Two rules for production:
- NEVER refit the K-means centers on updates — assign new patents to the
  frozen 500 centers, or the categories silently drift month to month.
- The signal for month t uses data through the last trading day of t and is
  tradeable at the t+1 open.

Keyless alternative: Google Patents Public Data on BigQuery (free sandbox
tier, Google account only, refreshed periodically) — same fields, different
plumbing.

## What we used instead of CRSP/Compustat (and what it costs us)

| Variable | Gold standard | What we use | Limitation |
|---|---|---|---|
| Monthly returns | CRSP (incl. delisting returns) | Yahoo Finance adjusted close, month-end to month-end | survivorship bias: delisted firms vanish from Yahoo; spreads likely understated |
| Excess return | CRSP − 1-mo T-bill | Yahoo return − Ken French RF series | none material |
| LnMCap | CRSP price x historical shares | adjusted close x CURRENT shares outstanding snapshot | buybacks/issuance history distorts old market caps; splits are handled (adjusted prices) |
| B/M | Compustat book common equity (+ deferred taxes − preferred) / Dec market cap | SEC EDGAR XBRL `StockholdersEquity` (as filed, point-in-time by FILING date) / same-month mcap | raw equity, no preferred-stock adjustment; ~65% firm-month coverage (IFRS-tag ADR filers missing) |
| ROE | Compustat quarterly NI / equity | EDGAR XBRL annual `NetIncomeLoss` / `StockholdersEquity`, point-in-time | annual not quarterly — staler between 10-Ks |
| Volatility (VOL) | daily CRSP | std of 20 trailing daily Yahoo returns | equivalent |
| Turnover (TO) | volume / CRSP shares | Yahoo volume / current shares snapshot | same shares caveat as LnMCap |
| Reversal (REV) | month t return | same | equivalent |
| Industry FE | GICS / FF48 | 2-digit SIC from EDGAR submissions | coarser than GICS; fine for FE |
| Risk-free | 1-mo T-bill | Ken French library RF | identical source |

Point-in-time discipline: fundamentals enter at their FILING date, not their
fiscal date — at month t the regression sees only what the market had by t.
This matches the paper's PIT principle and avoids look-ahead bias.

Upgrade path (in value order):
1. CRSP monthly file (via any university WRDS account) -> kills survivorship
   bias and fixes historical shares/market caps. Drop-in replacement for
   stage 3.
2. Compustat -> proper book equity + quarterly ROE. Replaces stage 3b.
3. KPSS patent-CRSP crosswalk -> validates/extends the stage-2 name match
   pre-2020. DISCERN adds dynamic ownership (M&A) through ~2015.
4. Exhibit-21 parsing from EDGAR -> systematic subsidiary->parent map
   (the DIY QuantData; replaces the hand-curated alias table).

## Research roadmap (beyond replication)

- [x] v1: bge-small, K=500, full tests (done — see results/)
- [ ] v2: bge-large 1024-dim embeddings, same downstream (running)
- [ ] K-robustness: rerun 5-8 at K=250 and K=1000
- [ ] Value-weighted results with CRSP-quality shares
- [ ] Inattention splits (analyst coverage via I/B/E/S or free proxies)
- [ ] Signal decay deep-dive: rolling-window coefficients year by year
```
