"""Central configuration for the technological-momentum pipeline.

Replicates Luo, Shi, Zhao, Wu & Geng (2025), "Technological Momentum in China"
(SSRN 5249018), adapted to US data per the original US antecedent
Lee, Sun, Wang & Zhang (2019, JFE), using free sources:

  patents   : USPTO PatentsView bulk downloads (no API key needed)
  firm link : SEC company_tickers.json, matched on normalized assignee name
  prices    : Yahoo Finance via yfinance (monthly returns)
  embeddings: BAAI/bge-*-en-v1.5 via sentence-transformers (English analogue
              of the paper's bge-large-zh-v1.5)
"""

import os
from pathlib import Path

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parent
DATA_RAW = ROOT / "data" / "raw"            # bulk zips as downloaded
DATA_INTERIM = ROOT / "data" / "interim"    # cleaned/matched tables
DATA_PROCESSED = ROOT / "data" / "processed"  # embeddings, vectors, signals
RESULTS = ROOT / "results"
SAMPLE_DIR = ROOT / "data" / "sample"       # synthetic data for smoke tests

for _d in (DATA_RAW, DATA_INTERIM, DATA_PROCESSED, RESULTS, SAMPLE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- sample window
# Patents granted on/after this date are ingested. The signal needs a trailing
# 12-month window, so the first usable signal month is one year later.
PATENT_START = "2009-01-01"
SIGNAL_START = "2010-01-31"     # first month-end for firm vectors / TECHMOM
SIGNAL_END = "2024-12-31"       # final PatentsView release covers to 12/31/2024
RETURNS_END = "2025-06-30"      # returns extend past SIGNAL_END for t+1 tests

# ---------------------------------------------------------------- PatentsView
# PatentsView migrated to the USPTO Open Data Portal (data.uspto.gov) in
# March 2026; the old S3 bucket now returns 403 and the ODP API needs a key.
# We use the official USPTO final-release mirror on Zenodo (CC-BY-4.0,
# data through 12/31/2024, direct HTTP, no key):
# https://zenodo.org/records/15783125
PATENTSVIEW_BASE = "https://zenodo.org/records/15783125/files"
PATENTSVIEW_TABLES = [
    "g_patent",                  # patent_id, patent_date, type (no abstract!)
    "g_patent_abstract",         # patent_id -> abstract text (separate table)
    "g_assignee_disambiguated",  # patent_id -> disambiguated assignee org
    "g_cpc_current",             # patent_id -> CPC section/class/subclass
]

SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
# SEC requires a descriptive User-Agent with contact info on all requests.
SEC_USER_AGENT = "Chicago Global research aliia.bekmagambetova@gmail.com"

# ---------------------------------------------------------------- embeddings
# bge-small-en-v1.5 (384-dim) runs ~10x faster than bge-large-en-v1.5
# (1024-dim, the faithful analogue of the paper's model) on CPU. Start small,
# switch to large for the final run if a GPU / enough hours are available.
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_BATCH_SIZE = 128           # RTX 5060 handles this; fine on CPU too
EMBED_SHARD_SIZE = 100_000       # abstracts per .npy shard (resumable)

# ---------------------------------------------------------------- clustering
N_CLUSTERS = 500                 # paper: K = 500 (vs 651 IPC subclasses)
CLUSTER_FIT_SAMPLE = 100_000     # paper: K-means fit on 100k random patents
SEED = 42

# ---------------------------------------------------------------- live updates
# Post-2024 patent data requires a (free, ID-verified) USPTO Open Data
# Portal API key. A HUMAN inserts it in ONE of two ways:
#   1. set environment variable  USPTO_ODP_API_KEY
#   2. paste the key into a file named  .odp_api_key  next to this config
#      (the file is gitignored — the key never enters version control)
# The live-update stage reads ODP_API_KEY; while it is None, the pipeline
# runs on the frozen 12/31/2024 Zenodo release only. See ROADMAP.md.
_key_file = ROOT / ".odp_api_key"
ODP_API_KEY = os.environ.get("USPTO_ODP_API_KEY") or (
    _key_file.read_text(encoding="utf-8").strip() if _key_file.exists() else None
)

# ---------------------------------------------------------------- signal/tests
MIN_FIRMS_PER_MONTH = 30         # skip months with too few firms with vectors
N_PORTFOLIOS = 5                 # quintile sorts
NEWEY_WEST_LAGS = 3              # paper: NW t-stats up to 3 lags
