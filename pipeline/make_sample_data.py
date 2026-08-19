"""Generate a small synthetic dataset to smoke-test stages 04-08 offline.

Simulates 60 firms in 8 technology areas, each filing patents whose
abstracts are drawn from area-specific templates, plus monthly returns
driven by persistent (AR(1)) area factors — so technologically linked
firms genuinely co-move with a lag and TECHMOM should show up positive
in the tests. Writes files with the same schemas the real stages emit:

  data/sample/patents.parquet     patent_id, ticker, grant_date, abstract
  data/sample/patent_cpc.parquet  patent_id, cpc_subclass
  data/sample/returns.parquet     month, ticker, ret, ln_mcap, vol, to, rev

Run:  python pipeline/make_sample_data.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

TOPICS = {
    "battery": ("H01M", ["lithium ion battery cell with improved {a} electrode",
                         "solid state electrolyte enabling higher {a} density",
                         "battery pack thermal management using {a} cooling"]),
    "display": ("G09F", ["display panel with {a} splicing frame for seamless tiling",
                         "oled pixel driving circuit reducing {a} artifacts",
                         "flexible display substrate with {a} encapsulation"]),
    "wireless": ("H04W", ["beamforming method for {a} millimeter wave links",
                          "handover protocol reducing {a} latency in 5g networks",
                          "antenna array with {a} interference cancellation"]),
    "genomics": ("C12Q", ["sequencing assay detecting {a} gene variants",
                          "crispr guide rna design for {a} editing efficiency",
                          "pcr amplification method with {a} fidelity"]),
    "engines": ("F02D", ["fuel injection control improving {a} combustion",
                         "turbocharger wastegate with {a} response",
                         "exhaust recirculation valve reducing {a} emissions"]),
    "semis": ("H01L", ["finfet transistor with {a} gate stack",
                       "wafer bonding process achieving {a} yield",
                       "interconnect barrier layer with {a} resistivity"]),
    "payments": ("G06Q", ["tokenized payment authorization with {a} fraud scoring",
                          "distributed ledger settlement reducing {a} latency",
                          "mobile wallet interface with {a} authentication"]),
    "imaging": ("G06T", ["image segmentation network with {a} attention",
                         "medical scan reconstruction using {a} priors",
                         "video compression exploiting {a} motion estimation"]),
}
ADJ = ["adaptive", "layered", "low-power", "high-precision", "modular",
       "robust", "scalable", "hybrid", "self-calibrating", "optimized"]

N_FIRMS = 60
PATENTS_PER_YEAR = 5
FACTOR_AR = 0.5          # persistence of area factors -> lagged co-movement
FACTOR_VOL = 0.04
IDIO_VOL = 0.06


def main() -> None:
    rng = np.random.default_rng(config.SEED)
    topics = list(TOPICS)
    firms = [f"FIRM{i:02d}" for i in range(N_FIRMS)]
    firm_topic = {f: topics[i % len(topics)] for i, f in enumerate(firms)}

    # ---- patents
    rows = []
    pid = 10_000_000
    dates = pd.date_range(config.PATENT_START, config.SIGNAL_END, freq="D")
    for firm in firms:
        main_topic = firm_topic[firm]
        n = int(PATENTS_PER_YEAR * len(dates) / 365)
        for _ in range(n):
            # 80% of filings in the firm's main area, 20% spillover
            topic = main_topic if rng.random() < 0.8 else rng.choice(topics)
            cpc, templates = TOPICS[topic]
            text = rng.choice(templates).format(a=rng.choice(ADJ))
            rows.append({
                "patent_id": str(pid),
                "ticker": firm,
                "grant_date": rng.choice(dates),
                "abstract": text,
                "cpc_subclass": cpc,
            })
            pid += 1
    patents = pd.DataFrame(rows)
    patents[["patent_id", "ticker", "grant_date", "abstract"]].to_parquet(
        config.SAMPLE_DIR / "patents.parquet", index=False)
    patents[["patent_id", "cpc_subclass"]].to_parquet(
        config.SAMPLE_DIR / "patent_cpc.parquet", index=False)

    # ---- returns: AR(1) area factors + idiosyncratic noise
    months = pd.date_range(config.SIGNAL_START, config.SIGNAL_END, freq="ME")
    factors = pd.DataFrame(index=months, columns=topics, dtype=float)
    prev = np.zeros(len(topics))
    for m in months:
        prev = FACTOR_AR * prev + rng.normal(0, FACTOR_VOL, len(topics))
        factors.loc[m] = prev
    ret_rows = []
    for firm in firms:
        f = factors[firm_topic[firm]].to_numpy()
        r = f + rng.normal(0, IDIO_VOL, len(months))
        ret_rows.append(pd.DataFrame({
            "month": months, "ticker": firm, "ret": r,
            "ln_mcap": rng.normal(22, 1),
            "vol": np.abs(rng.normal(0.02, 0.005, len(months))),
            "to": np.abs(rng.normal(1.5, 0.5, len(months))),
        }))
    returns = pd.concat(ret_rows, ignore_index=True)
    returns["rev"] = returns["ret"]
    returns.to_parquet(config.SAMPLE_DIR / "returns.parquet", index=False)

    print(f"sample: {len(patents):,} patents, {N_FIRMS} firms, "
          f"{len(months)} months -> {config.SAMPLE_DIR}")


if __name__ == "__main__":
    main()
