# Technological Momentum — US Replication Pipeline

Replicates the data pipeline of **Luo, Shi, Zhao, Wu & Geng (2025), "Technological
Momentum in China: Large Language Model Meets Simple Classifications"** (SSRN 5249018),
adapted to **US data** following the paper's US antecedent, Lee, Sun, Wang & Zhang
(2019, *JFE*), "Technological Links and Predictable Returns".

> **Start here** if you just want to run the pipeline — everything needed is on
> this page. For the internals — stage dependency tables, restart rules, the
> live-data API key procedure, data-source substitutions, and the research
> roadmap — see **[ROADMAP.md](ROADMAP.md)**. For the empirical findings —
> all specifications, tables, and interpretation — see **[RESULTS.md](RESULTS.md)**.

## The idea

Firms whose patents describe similar technology are "technology-linked". When a firm's
tech peers do well this month, the firm's own stock tends to drift in the same
direction next month — a cross-firm momentum spillover. The paper measures the links
two ways and compares them:

- **Classification-based** (`techmom_cls`): each patent tagged by its patent-office
  subclass (paper: IPC, 651 subclasses; here: CPC subclass from PatentsView).
- **LLM-based** (`techmom_bge`): each patent abstract embedded with a BGE model,
  K-means clustered into 500 "technology themes" (fit on a 100k random sample).

For each firm-month: count new patents over the trailing 12 months per category
→ firm vector `c_it` → `LINK_ijt = cosine(c_it, c_jt)` →
`TECHMOM_it = Σ_j LINK_ijt · RET_jt / Σ_j LINK_ijt`. Signal at month *t* predicts
returns at *t+1* (Fama-MacBeth + orthogonalized quintile sorts, Newey-West t-stats).

## Data sources (all free)

| Paper (China) | This pipeline (US) |
|---|---|
| QuantData patent texts, subsidiary-consolidated | USPTO **PatentsView** bulk files (`g_patent`, `g_assignee_disambiguated`, `g_cpc_current`) |
| bge-large-zh-v1.5 embeddings | `BAAI/bge-small-en-v1.5` (switch to `bge-large-en-v1.5` in `config.py` for the faithful version) |
| IPC subclasses (651) | CPC subclasses (same 4-char granularity) |
| RiceQuant / Tushare stock data | SEC `company_tickers.json` for the firm link + Yahoo Finance monthly returns |

## Running

```bash
# smoke test on synthetic data (~5 min, validates every stage end-to-end)
python run_pipeline.py --sample

# full run (roughly: 1h download + 2-6h embedding on CPU + 1h rest)
python run_pipeline.py

# rerun individual stages (embedding is sharded and resumable)
python run_pipeline.py --stages 4 5
```

Stages (`pipeline/`):

1. **download** — PatentsView bulk zips + SEC ticker file (resumable)
2. **link_firms** — normalized-name match of patent assignees to SEC registrants
3. **returns** — Yahoo daily prices → monthly returns, ln(mcap), vol, turnover, reversal
4. **embed** — abstracts → L2-normalized embeddings (sharded, resumable)
5. **cluster** — MiniBatchKMeans, K=500 (paper convention; SSE-vs-K is a
   scale-free power law, so no K is data-privileged — see RESULTS §4 and the
   K-free continuous variant)
6. **firm_vectors** — trailing-12m patent counts per category, per firm-month
7. **techmom** — cosine LINK matrix + similarity-weighted peer returns
8. **tests** — Fama-MacBeth and quintile portfolio sorts → `results/*.csv`

## Step-by-step for someone who has never used a terminal (Windows)

Every command in this README is typed into **PowerShell** — Windows' built-in
command window. (On Mac/Linux, use the Terminal app instead; commands are the
same except paths use `/`.)

1. **Open PowerShell**: press the Windows key, type `powershell`, press Enter.
   A blue/black window appears with a blinking cursor. You type commands there
   and press Enter to run them.
2. **Go to the project folder** (adjust the path to where the folder lives):
   ```powershell
   cd "C:\Users\<you>\Desktop\Chicago Global\tech-momentum"
   ```
3. **Check Python is installed**:
   ```powershell
   python --version
   ```
   You want to see `Python 3.12.x` (3.11+ is fine). If instead you get an
   error, or a Microsoft Store window opens: install Python from
   https://www.python.org/downloads/ and — IMPORTANT — tick the
   **"Add python.exe to PATH"** checkbox in the installer, then close and
   reopen PowerShell and try again.
4. **Install the dependencies** (one-time, ~5 min):
   ```powershell
   python -m pip install torch --index-url https://download.pytorch.org/whl/cu128   # NVIDIA GPU
   # ...or, if the machine has no NVIDIA GPU:  python -m pip install torch
   python -m pip install -r requirements.txt
   ```
5. **Run the 5-minute smoke test** — always do this first:
   ```powershell
   python run_pipeline.py --sample
   ```
   If it ends with result tables printed, the machine is set up correctly.
6. **Run the real pipeline** (hours; safe to leave unattended, and every
   heavy stage resumes where it stopped if interrupted):
   ```powershell
   python run_pipeline.py
   ```
   Results land in the `results\` folder as CSV files (open in Excel).

Useful variations: `python run_pipeline.py --list` shows all stages;
`--only tests` reruns just the statistics; `--from cluster` reruns from
clustering onward; `--skip download` skips the download.

## Reproducing from scratch (for a new user)

```bash
git clone <this repo>  &&  cd tech-momentum
python -m venv .venv && .venv/Scripts/activate      # or source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu128   # GPU; or plain `pip install torch` for CPU
pip install -r requirements.txt

python run_pipeline.py --sample     # 5-min smoke test: validates every stage
python run_pipeline.py              # full run (see runtimes above)
```

Determinism: all sampling and K-means use `SEED` from `config.py`, and the patent
source is a DOI-frozen Zenodo release — two users get identical clusters and
signals. Prices (Yahoo) and fundamentals (EDGAR) are live sources and may drift
marginally between runs.

## Keeping the signal up to date (production mode)

The Zenodo snapshot is frozen at 12/31/2024. For a live monthly signal:

1. Register a free API key for the USPTO Open Data Portal (data.uspto.gov) —
   PatentsView's successor, updated regularly. This is the only step that
   needs a human (account creation).
2. Monthly job: pull patents granted since the last snapshot (grant text +
   assignee + CPC), append to `data/raw`, rerun stages 2 and 4 — embedding is
   incremental, so only new patents are embedded.
3. Keep the K-means cluster centers FROZEN from the original fit; assign new
   patents to existing centers. Refitting monthly would silently redefine the
   technology categories and make the signal non-comparable over time.
4. Rerun stages 3/3b (prices, fundamentals refresh) and 6-7 for the new
   month-end. TECHMOM for month t is available on the last trading day of t —
   tradeable at the t+1 open.

## Known limitations vs the paper (talking points)

- **Subsidiary consolidation**: QuantData maps patents to ultimate parents (~60% of
  Chinese listed-firm patents sit in subsidiaries). Our name match catches only
  assignees named like the listed parent. Fix: KPSS/NBER assignee-CRSP crosswalks.
- **Fundamentals controls**: BM and ROE need point-in-time Compustat; Yahoo snapshots
  are not point-in-time, so those controls (and industry fixed effects) are omitted.
- **Survivorship**: Yahoo drops delisted tickers, biasing returns up. Fix: CRSP.
- **Shares outstanding** is a current snapshot, so mcap/turnover are approximations.
- Value-weighted sorts and the investor-inattention split (media/analyst coverage)
  need data we don't have for free; equal-weighted sorts are reported.
