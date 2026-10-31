import numpy as np
import pandas as pd
import pytest
from scipy import stats as st
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportions_ztest

from pulsescope import stats as S


def test_two_proportion_matches_statsmodels():
    r = S.two_proportion_ztest(300, 1000, 345, 1000)
    z, p = proportions_ztest([345, 300], [1000, 1000])
    assert r["z"] == pytest.approx(z)
    assert r["p_value"] == pytest.approx(p)
    assert r["diff"] == pytest.approx(0.045)


def test_welch_matches_scipy():
    rng = np.random.default_rng(0)
    a, b = rng.normal(10, 2, 500), rng.normal(10.4, 3, 700)
    r = S.welch_ttest(a, b)
    ref = st.ttest_ind(b, a, equal_var=False)
    assert r["t"] == pytest.approx(ref.statistic)
    assert r["p_value"] == pytest.approx(ref.pvalue)
    lo, hi = r["diff_ci"]
    assert lo < r["diff"] < hi


def test_benjamini_hochberg_matches_statsmodels():
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.5, 0.9])
    assert np.allclose(S.benjamini_hochberg(p), multipletests(p, method="fdr_bh")[1])


def test_wilson_ci_contains_estimate_and_is_bounded():
    p, lo, hi = S.wilson_ci(0, 50)
    assert lo == pytest.approx(0, abs=1e-12) and 0 < hi < 0.1
    p, lo, hi = S.wilson_ci(37, 100)
    assert lo < p < hi


def test_chi2_independence_detects_dependence():
    df = pd.DataFrame({"g": ["a"] * 500 + ["b"] * 500, "y": [1] * 300 + [0] * 200 + [1] * 200 + [0] * 300})
    assert S.chi2_independence(df, "g", "y")["p_value"] < 1e-6


def test_robust_z_flags_outlier():
    x = np.r_[np.random.default_rng(1).normal(0, 1, 100), 15]
    assert abs(S.robust_z(x)[-1]) > 8
