"""Invariant tests — assertions on the real data artifacts.

These encode facts that CANNOT be false if the pipeline is healthy. They are
the checks whose absence let impossible numbers (a value-weighted book of
large US patent holders losing 0.5%/month while the index quadrupled) reach
the repository unnoticed.

Each test skips if its artifact does not exist yet, so the suite is usable
at any pipeline stage:   python -m pytest tests/ -v
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config

RET = config.DATA_INTERIM / "returns.parquet"
TM = config.DATA_PROCESSED / "techmom.parquet"
EMB = config.DATA_PROCESSED / "embeddings"
CL = config.DATA_PROCESSED / "patent_clusters.parquet"
PAT = config.DATA_INTERIM / "patents.parquet"


def need(path: Path):
    if not path.exists():
        pytest.skip(f"{path.name} not built yet")


class TestReturnsPanel:
    def test_returns_finite(self):
        need(RET)
        r = pd.read_parquet(RET, columns=["ret"])
        assert np.isfinite(r["ret"]).all(), "non-finite returns in panel"

    def test_no_impossible_market_caps(self):
        need(RET)
        r = pd.read_parquet(RET, columns=["ln_mcap"]).dropna()
        assert np.exp(r["ln_mcap"].max()) < 6e12, \
            "implied market cap above $6tn — corrupted shares/price data"

    def test_value_weighted_panel_return_plausible(self):
        """Ben's assertion: a cap-weighted book of large US patent holders
        must sit within a plausible band of the market benchmark."""
        need(RET)
        r = pd.read_parquet(RET).dropna(subset=["ret", "ln_mcap"])
        r["month"] = pd.to_datetime(r["month"])
        r = r[r["month"] >= "2010-01-01"]
        r["mcap"] = np.exp(r["ln_mcap"])
        vw = (r.groupby("month")
              .apply(lambda g: np.average(g["ret"], weights=g["mcap"]),
                     include_groups=False).mean())
        assert 0.002 < vw < 0.03, \
            f"panel VW mean {vw:+.4f}/mo outside plausible band [0.2%, 3%]"

    def test_biggest_weights_are_real_megacaps(self):
        need(RET)
        r = pd.read_parquet(RET).dropna(subset=["ln_mcap"])
        r["mcap"] = np.exp(r["ln_mcap"])
        r["w"] = r["mcap"] / r.groupby("month")["mcap"].transform("sum")
        top5 = set(r.groupby("ticker")["w"].mean().nlargest(5).index)
        known = {"AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSM",
                 "TM", "XOM", "JNJ", "BRK-B", "TSLA"}
        impostors = top5 - known
        assert not impostors, f"unknown firms among top value weights: {impostors}"


class TestSignal:
    def test_techmom_bounded(self):
        need(TM)
        t = pd.read_parquet(TM)
        for col in ("techmom_bge", "techmom_cls"):
            s = t[col].dropna()
            s = s[np.isfinite(s)]
            # a similarity-weighted average of monthly returns should rarely
            # leave (-75%, +150%); far outside means corrupted peer returns
            assert s.quantile(0.001) > -0.75 and s.quantile(0.999) < 1.5, \
                f"{col} tails imply corrupted peer returns"

    def test_signal_month_precedes_return_month(self):
        """Timing: TECHMOM at month t must exist before the t+1 returns it
        predicts — max signal month <= max return month - 0 (same-month
        construction, forward merge does the shift)."""
        need(TM); need(RET)
        tmax = pd.to_datetime(pd.read_parquet(TM, columns=["month"])["month"]).max()
        rmax = pd.to_datetime(pd.read_parquet(RET, columns=["month"])["month"]).max()
        assert tmax <= rmax, "signal months extend beyond return data"


class TestEmbeddings:
    def test_unit_norm(self):
        if not (EMB / "shard_0000.npy").exists():
            pytest.skip("no embeddings yet")
        x = np.load(EMB / "shard_0000.npy")
        norms = np.linalg.norm(x[:1000], axis=1)
        assert np.allclose(norms, 1.0, atol=1e-3), "embeddings not L2-normalized"

    def test_index_matches_shards(self):
        if not (EMB / "index.parquet").exists():
            pytest.skip("no embedding index yet")
        idx = pd.read_parquet(EMB / "index.parquet")
        n_rows = sum(np.load(EMB / f"shard_{s:04d}.npy", mmap_mode="r").shape[0]
                     for s in sorted(idx["shard"].unique()))
        assert n_rows == len(idx), "embedding index out of sync with shards"
        assert idx["patent_id"].is_unique, "duplicate patents in embedding index"


class TestClusters:
    def test_cluster_range_and_coverage(self):
        need(CL); need(PAT)
        cl = pd.read_parquet(CL)
        assert cl["cluster"].between(0, config.N_CLUSTERS - 1).all()
        n_pat = len(pd.read_parquet(PAT, columns=["patent_id"]))
        assert len(cl) >= 0.95 * n_pat, "many linked patents lack a cluster"
        assert cl["cluster"].nunique() > config.N_CLUSTERS * 0.5, \
            "less than half the clusters occupied — degenerate clustering"
