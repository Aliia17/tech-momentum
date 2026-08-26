"""Unit tests — pure-function correctness, no data files needed.

Run:  python -m pytest tests/ -v
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "pipeline"))

from stage02_link_firms import ALIASES, normalize
from stage07_techmom import techmom_one_month
from stage08_tests import nw_tstat, winsorize
import config


class TestNormalize:
    def test_strips_legal_suffixes(self):
        assert normalize("Apple Inc.") == "apple"
        assert normalize("NVIDIA CORPORATION") == "nvidia"

    def test_strips_state_of_incorporation_tail(self):
        assert normalize("QUALCOMM INC/DE") == "qualcomm"
        assert normalize("CORNING INC /NY") == "corning"

    def test_strips_leading_the(self):
        assert normalize("The Boeing Company") == "boeing"

    def test_ampersand(self):
        assert normalize("Procter & Gamble Co") == "procter and gamble"

    def test_no_substring_matching_risk(self):
        # a subsidiary-like name must NOT normalize to its parent's name
        assert normalize("Microsoft Ireland Operations Ltd") != "microsoft"

    def test_alias_keys_are_normalized_forms(self):
        for key in ALIASES:
            assert key == normalize(key), f"alias key not normal-form: {key}"


class TestWinsorize:
    def test_bounds(self):
        s = pd.Series(np.concatenate([[1e9, -1e9], np.random.default_rng(0)
                                      .normal(0, 1, 1000)]))
        w = winsorize(s)
        assert w.max() <= s.quantile(0.99) + 1e-9
        assert w.min() >= s.quantile(0.01) - 1e-9

    def test_clips_infinity(self):
        s = pd.Series([np.inf, -np.inf] + list(range(200)))
        assert np.isfinite(winsorize(s)).all()


class TestTechmom:
    def test_hand_computed_example(self):
        """3 firms, 2 tech dims; verify eq. (2)-(3) by hand.

        A=[1,0], B=[0,1], C=[1,1]:
          cos(A,B)=0, cos(A,C)=cos(B,C)=1/sqrt(2)
          TECHMOM_A = ret_C (C is A's only linked peer)
          TECHMOM_C = (ret_A+ret_B)/2 (equal similarity to both)
        """
        vecs = pd.DataFrame({
            "ticker": ["A", "B", "C", "C"],
            "dim": ["d1", "d2", "d1", "d2"],
            "count": [1, 1, 1, 1],
        })
        rets = pd.Series({"A": 0.10, "B": -0.02, "C": 0.04})
        old_min = config.MIN_FIRMS_PER_MONTH
        config.MIN_FIRMS_PER_MONTH = 2
        try:
            tm = techmom_one_month(vecs, rets)
        finally:
            config.MIN_FIRMS_PER_MONTH = old_min
        assert tm["A"] == pytest.approx(0.04)
        assert tm["B"] == pytest.approx(0.04)
        assert tm["C"] == pytest.approx((0.10 - 0.02) / 2)

    def test_self_excluded(self):
        """A firm's own return must not enter its TECHMOM."""
        vecs = pd.DataFrame({"ticker": ["A", "B"], "dim": ["d1", "d1"],
                             "count": [1, 1]})
        rets = pd.Series({"A": 1.00, "B": 0.00})
        old_min = config.MIN_FIRMS_PER_MONTH
        config.MIN_FIRMS_PER_MONTH = 2
        try:
            tm = techmom_one_month(vecs, rets)
        finally:
            config.MIN_FIRMS_PER_MONTH = old_min
        assert tm["A"] == pytest.approx(0.00)   # only peer B's return


class TestNeweyWest:
    def test_constant_series_recovers_mean(self):
        mean, t = nw_tstat(pd.Series([0.01] * 60), lags=3)
        assert mean == pytest.approx(0.01)

    def test_zero_mean_not_significant(self):
        rng = np.random.default_rng(1)
        mean, t = nw_tstat(pd.Series(rng.normal(0, 0.05, 240)), lags=3)
        assert abs(t) < 3
