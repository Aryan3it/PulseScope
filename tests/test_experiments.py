import numpy as np
import pandas as pd
import pytest

from pulsescope import experiments as E


def test_sample_size_reasonable():
    # 10% baseline, 10% relative MDE -> ~14.7k per arm (standard calculator value)
    n = E.sample_size_proportion(0.10, 0.10)
    assert 14_000 < n < 15_500


def test_srm_detects_imbalance_and_passes_balance():
    assert E.srm_test(50_000, 50_100)["srm_detected"] is False
    assert E.srm_test(50_000, 48_000)["srm_detected"] is True


def test_cuped_reduces_variance_without_bias():
    rng = np.random.default_rng(0)
    n = 20_000
    x = rng.gamma(2, 50, n)
    variant = np.where(rng.random(n) < 0.5, "treatment", "control")
    y = 0.8 * x + rng.normal(0, 20, n) + np.where(variant == "treatment", 5.0, 0.0)
    r = E.cuped(pd.DataFrame({"y": y, "x": x, "variant": variant}), "y", "x")
    assert r["variance_reduction"] > 0.8
    assert r["cuped"]["diff"] == pytest.approx(5.0, abs=1.0)
    assert r["cuped"]["se"] < r["raw"]["se"]


def test_obf_boundaries_match_published_values():
    # Five equally spaced looks, alpha=0.05 two-sided: OBF final boundary ~2.04, first ~4.56.
    b = E.obf_boundaries(np.arange(1, 6) / 5)
    assert b[-1] == pytest.approx(2.04, abs=0.03)
    assert b[0] == pytest.approx(4.56, abs=0.07)


def test_peeking_inflates_fpr_and_obf_controls_it():
    r = E.peeking_simulation(n_sim=3000)
    assert r["naive_fpr"] > 0.12
    assert r["obf_fpr"] == pytest.approx(0.05, abs=0.015)


def test_bootstrap_ci_covers_truth():
    rng = np.random.default_rng(1)
    a, b = rng.exponential(100, 5000), rng.exponential(110, 5000)
    r = E.bootstrap_diff_ci(a, b, n_boot=500)
    assert r["ci"][0] < 0.10 < r["ci"][1]
