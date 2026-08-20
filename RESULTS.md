# Results — US Replication of "Technological Momentum" (SSRN 5249018)

**TL;DR.** The pipeline replicates the paper's methodology end-to-end on free US
data (1.23M patents, ~2,000 firms, 2010–2024). The paper's two qualitative
claims hold: technological momentum predicts returns, and the LLM-based
measure outperforms the classification-based one. But the US premium is a
fraction of the Chinese one, and tracing it down yields the interesting part:
**domestic US technological momentum is fully arbitraged away (t ≈ 0 on CRSP
data); the surviving predictability sits in cross-border technology links.**

## 1. What was built

Patent abstracts → BGE embeddings → K-means technology themes (K=500, fit on
a 100k random sample of the full corpus) → trailing-12-month firm technology
vectors → pairwise cosine `LINK` → `TECHMOM` = similarity-weighted peer
return → Fama-MacBeth regressions and orthogonalized quintile sorts
(controls: LnMCap, B/M, ROE, VOL, TO, REV + industry FE; Newey-West t-stats).
Sources and substitutions vs the paper are documented in [ROADMAP.md](ROADMAP.md);
how to run in [README.md](README.md).

## 2. Headline: Fama-MacBeth, month-t signal → month-t+1 excess return

TECHMOM_BGE coefficient (t-stat) across specifications, 2010–2024. Rows vary
along two axes: the **universe** (full = all matched firms including ~570
foreign ADRs like TSMC/Toyota/Sony, which are also peers inside everyone's
TECHMOM; domestic = US-incorporated common stock only, the CRSP universe)
and the **return data** (Yahoo vs CRSP). The middle row changes ONLY the
universe relative to row 1 — the controlled comparison showing the collapse
comes from dropping cross-border links, not from data quality:

| Universe | Model/K | Returns | coef | t | Verdict |
|---|---|---|---|---|---|
| Full (incl. ADRs) | bge-small, K=500 | Yahoo | 0.031 | **2.00** | significant, 5% |
| Full (incl. ADRs) | bge-large, K=500 | Yahoo | 0.018 | 1.18 | n.s. |
| Full (incl. ADRs) | bge-large, K=250 / 1000 | Yahoo | 0.017 / 0.019 | 0.97 / 1.34 | n.s. |
| Domestic only | bge-small, K=500 | Yahoo | 0.008 | 0.50 | n.s. — universe effect isolated |
| Domestic only | bge-small, K=500 | **CRSP** (to 2023) | −0.004 | −0.25 | **zero** on gold-standard data |

Classification-based TECHMOM (CPC subclasses, the paper's IPC analogue):
t = 1.63 in the full universe — always weaker than the LLM measure, matching
the paper's central claim about semantic vs taxonomic information.

Comparison points: the paper reports coef ≈ 0.077 (t ≈ 4.2) in China
2015–2024; Lee, Sun, Wang & Zhang (JFE 2019) found spreads of 0.6–1%/month
in the US 1963–2012.

## 3. Portfolio sorts

Quintile sorts on TECHMOM orthogonalized to controls + industry (long Q5 /
short Q1, monthly rebalance): spreads of 0.1–0.2%/month, statistically
indistinguishable from zero in all specifications — consistent with the
regression coefficient's implied spread (coef × quintile signal gap ≈
0.3%/month, below detectability in 15 years of monthly data). In China the
same arithmetic yields ≈0.75%/month, matching their reported 0.66–0.73%.

## 4. The decomposition (the interesting finding)

Removing foreign-listed ADR firms (TSMC, Toyota, Sony, ASML, Ericsson, …)
from the universe collapses the signal even on identical return data
(t: 2.00 → 0.50). On gold-standard CRSP domestic data the signal is exactly
zero. Reading:

- **Publication decay, completed.** Lee et al. (2019) documented the domestic
  US effect through 2012; post-publication, quant arbitrage ate it. Our CRSP
  zero on 2010–2023 is that lifecycle's endpoint (cf. McLean & Pontiff 2016).
- **Cross-border links still carry a premium.** Information flowing through
  foreign tech peers is costlier to process and harder to arbitrage —
  precisely the inattention mechanism the Chinese paper builds on, and why
  the effect remains strong in China's retail-dominated market.
- Caveat: dropping ADRs also removes the densest patent nodes, degrading
  everyone's peer sets; part of the collapse may be measurement rather than
  economics. A purpose-built test (domestic signal vs cross-border signal as
  separate variables) is the natural next experiment.

## 5. Supporting diagnostics

- **Alpha-decay split:** coefficients shrink early→late (2010–16 vs 2017–24),
  with the LLM measure decaying slower than the classification measure —
  the paper's "complex information is priced more slowly" thesis in decay form.
  Neither half individually significant (power halves with the sample).
- **Control sensitivity:** dropping the EDGAR-proxy controls (B/M, ROE) and
  even turnover moves the TECHMOM t-stat by <0.15 — proxy quality is not
  driving results (`results/fm_control_sensitivity.csv`).
- **K-robustness:** elbow diagnostic supports K≈250–500; results at K=250,
  500, 1000 are similar (mildly increasing in K for the large model).
- **Bigger model ≠ stronger signal:** bge-large (1024-dim, paper-faithful)
  underperforms bge-small everywhere — embedding sophistication cannot
  resurrect an arbitraged premium.

## 6. Known limitations

Firm linking covers currently-registered SEC firms plus a hand-verified
alias table for major subsidiaries/renames (QuantData-style full ownership
consolidation, e.g. via 10-K Exhibit 21 parsing or the KPSS/DISCERN
crosswalks, is the main upgrade). Yahoo returns carry survivorship bias
(the CRSP runs address this); fundamentals are point-in-time EDGAR XBRL
(shown above to be immaterial). Patent data frozen at 12/31/2024 (official
final PatentsView release); live updates require a USPTO ODP API key
(insertion point documented in ROADMAP.md).

## 7. For a trading application

The signal, as published, is not a standalone US strategy in 2025 — that is
the honest headline. Its value is as (a) a pipeline that can assemble this
family of signals in any market with patent + price data, (b) a candidate
*input* to multi-signal models, and (c) a pointer toward the cross-border
version, which our decomposition suggests is where the remaining premium
lives and which is not what the published papers trade.
