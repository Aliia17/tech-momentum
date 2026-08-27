# Results — US Replication of "Technological Momentum" (SSRN 5249018)

> **Revision note.** These results supersede an earlier version of this file.
> An external review found impossible value-weighted portfolio returns in the
> committed results; the root cause (corrupted market caps and returns for
> serial reverse-split penny stocks in the free price feed) has been fixed,
> a test suite now guards the invariants, and every number below is from the
> repaired panel. Details in §7. The headline changed: specifications that
> previously appeared marginally significant are not, once the data is clean.

**TL;DR.** The pipeline replicates the paper's methodology end-to-end on free
US data (1.23M patents, ~2,000 firms, 2010–2024), including the
paper-faithful embedding model (bge-large, 1024-dim — the English sibling of
the paper's bge-large-zh). **In that headline specification the US signal is
indistinguishable from zero (t = 0.14)**; across all robustness variants
(smaller embedding model, different cluster counts, data sources, universes)
the t-statistic never exceeds 1.33 — nothing is significant on clean data.
Combined with the strong published results for 1963–2012 (Lee et al. 2019,
JFE) and for China 2015–2024 (the replicated paper), the picture is
coherent: **the US premium has decayed to statistical zero since
publication**, with weak directional hints that what little remains involves
cross-border technology links.

## 1. What was built

Patent abstracts → BGE embeddings → K-means technology themes (K=500, fit on
a 100k random sample of the full corpus) → trailing-12-month firm technology
vectors → pairwise cosine `LINK` → `TECHMOM` = similarity-weighted peer
return → Fama-MacBeth regressions and orthogonalized quintile sorts
(controls: LnMCap, B/M, ROE, VOL, TO, REV + industry FE; Newey-West t-stats).
Returns panel is CRSP-first (returns, market caps, turnover) with Yahoo
Finance filling only what CRSP cannot cover (foreign ADRs, post-2023), plus
quarantine rules for unverifiable data. Sources and substitutions:
[ROADMAP.md](ROADMAP.md); how to run: [README.md](README.md).

## 2. Headline: Fama-MacBeth, month-t signal → month-t+1 excess return

TECHMOM_BGE coefficient (t-stat), 2010–2024, repaired panel. The headline is
the paper-faithful specification: **bge-large (1024-dim), K=500, full
universe** — the same model family and size logic as the paper's
bge-large-zh. The faster bge-small variant is a robustness check, not the
headline:

| Spec | Universe | Model/K | coef | t | Verdict |
|---|---|---|---|---|---|
| **HEADLINE** | Full (incl. ADRs) | **bge-large, K=500** | 0.002 | **0.14** | zero |
| robustness | Full (incl. ADRs) | bge-small, K=500 | 0.025 | 1.33 | n.s. |
| robustness | Domestic US only | bge-small, K=500 (Yahoo) | 0.011 | 0.63 | n.s. |
| robustness | Domestic US only | bge-small, K=500 (**CRSP**, to 2023) | −0.004 | −0.25 | zero |

Classification-based TECHMOM (CPC subclasses): t = 0.44 in the headline
spec — weaker than the small-model LLM measure, consistent with the paper's
ordering, though on clean US data neither is distinguishable from zero.

Comparison points: the paper reports coef ≈ 0.077 (t ≈ 4.2) in China
2015–2024; Lee et al. (2019) found 0.6–1%/month spreads in the US 1963–2012.

## 3. Portfolio sorts

Quintile sorts on orthogonalized TECHMOM (long Q5 / short Q1, monthly):
spreads within ±0.23%/month, all |t| < 1.2, equal- and value-weighted.
Quintile *levels* are now sensible (~1%/month, consistent with the market),
confirming the repaired weights.

## 4. Supporting diagnostics (repaired panel; computed on the small-model
signal, where there is enough variation to diagnose)

- **Alpha decay:** early half (2010–16) t = 1.43 (BGE) / 1.81 (CLS); late
  half (2017–24) t = 0.47 / −0.74. The effect weakens to nothing in the
  post-publication era — the McLean-Pontiff pattern.
- **Cross-border hint:** full universe (with foreign ADR peers) t = 1.33 vs
  domestic-only t = 0.63 on identical data. Directionally consistent with
  residual predictability living in cross-border links, but the difference
  is not statistically established.
- **Control sensitivity:** dropping B/M+ROE or turnover moves the TECHMOM
  t by <0.15 — proxy quality of controls is immaterial.
- **K-robustness:** results similar at K = 250/500/1000.
- **Embedding-model sensitivity:** the paper-faithful large model (headline)
  shows less signal than the small robustness variant (t = 0.14 vs 1.33).
  That the result varies this much across reasonable embedding choices — and
  never reaches significance — is itself evidence of a premium that no
  longer exists in exploitable form.

## 5. Interpretation

The three-study arc is internally consistent: strong premium in the US when
patent data was effectively invisible to investors (1963–2012), strong
premium today in a retail-dominated, arbitrage-constrained market (China),
zero in the modern US where the mechanism has been public since 2019. Our
contribution is documenting the endpoint on clean data with a reproducible
pipeline. For a fund: the signal as published is not a US strategy in 2025;
its value is the pipeline (any market, any patent corpus), the candidate-
input role in multi-signal models, and the untested cross-border variant.

## 6. Known limitations

Firm linking covers currently-registered SEC firms plus a hand-verified
alias table (full ownership consolidation à la QuantData — Exhibit-21
parsing or KPSS/DISCERN — is the main upgrade). Non-CRSP firm-months rely
on Yahoo with quarantine rules. Fundamentals are point-in-time EDGAR XBRL
(shown immaterial). Patents frozen at 12/31/2024 (final PatentsView
release); the live-update key insertion point is documented in ROADMAP.md.

## 7. Post-review corrections and verification

An external review of an earlier version found value-weighted portfolios
losing ~0.5%/month over a period when the index quadrupled — impossible for
a cap-weighted book of large patent holders — and correctly diagnosed the
deeper issue: no tests, so impossible numbers could not fail.

**Root cause.** Market cap was computed as adjusted price × current shares.
For serial reverse-split penny stocks, Yahoo's adjustment inflates past
prices by the cumulative split factor; combined with current share counts
this manufactured fake mega-caps (one nano-cap carried an implied $316tn and
81% of the value-weighted book while losing 9%/month). The same firms'
return series were also corrupted, which had inflated the regression
results (headline t = 2.00 → 1.33 after repair).

**Fix** (`pipeline/stage03d_patch_mcap.py`): CRSP-first returns/caps/
turnover wherever CRSP covers the firm-month; quarantine of corrupt
non-CRSP tickers (reverse-split signatures, untradeable size); value
weights only for verifiable caps (CRSP or established-ADR whitelist);
finite-return enforcement.

**Verification** (`tests/`, run `python -m pytest tests/`): 21 tests —
unit tests (name normalization, hand-computed TECHMOM per eqs. (2)–(3),
self-exclusion, winsorization, Newey-West) and invariant tests on the real
artifacts, including the reviewer's assertion: any value-weighted panel
return must sit in a plausible band of the market benchmark; top value
weights must be recognizable mega-caps; returns finite; signal tails
bounded; embeddings unit-norm; timing correct. The suite failed on first
run (catching two further corrupt tickers and the contaminated peer
returns) and passes in full on the repaired data.
