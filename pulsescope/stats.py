"""Descriptive and inferential statistics primitives used across the project."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as st


def describe(x, name: str = "x") -> dict:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    q = np.percentile(x, [1, 5, 25, 50, 75, 95, 99])
    return {
        "metric": name, "n": int(len(x)), "mean": float(x.mean()), "std": float(x.std(ddof=1)),
        "sem": float(st.sem(x)), "min": float(x.min()), "p01": q[0], "p05": q[1], "p25": q[2], "median": q[3],
        "p75": q[4], "p95": q[5], "p99": q[6], "max": float(x.max()),
        "iqr": float(q[4] - q[2]), "cv": float(x.std(ddof=1) / x.mean()) if x.mean() else np.nan,
        "skew": float(st.skew(x)), "excess_kurtosis": float(st.kurtosis(x)),
        "share_zero": float((x == 0).mean()),
    }


def mean_ci(x, alpha: float = 0.05) -> tuple[float, float, float]:
    x = np.asarray(x, dtype=float)
    m, se = x.mean(), st.sem(x)
    h = se * st.t.ppf(1 - alpha / 2, len(x) - 1)
    return m, m - h, m + h


def wilson_ci(successes: int, n: int, alpha: float = 0.05) -> tuple[float, float, float]:
    z = st.norm.ppf(1 - alpha / 2)
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return p, centre - half, centre + half


def two_proportion_ztest(x1: int, n1: int, x2: int, n2: int, alpha: float = 0.05) -> dict:
    """Group 1 = control/baseline, group 2 = treatment/current. Pooled SE for the test, unpooled for the CI."""
    p1, p2 = x1 / n1, x2 / n2
    pooled = (x1 + x2) / (n1 + n2)
    se_pool = np.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    z = (p2 - p1) / se_pool if se_pool > 0 else 0.0
    se = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    zc = st.norm.ppf(1 - alpha / 2)
    # Relative lift CI via the delta method on log(p2/p1).
    se_log = np.sqrt((1 - p1) / (n1 * p1) + (1 - p2) / (n2 * p2)) if p1 > 0 and p2 > 0 else np.nan
    rel = p2 / p1 - 1 if p1 > 0 else np.nan
    return {
        "p_control": p1, "p_treatment": p2, "diff": p2 - p1,
        "diff_ci": (p2 - p1 - zc * se, p2 - p1 + zc * se),
        "rel_lift": rel,
        "rel_lift_ci": (np.exp(np.log(p2 / p1) - zc * se_log) - 1, np.exp(np.log(p2 / p1) + zc * se_log) - 1)
        if p1 > 0 and p2 > 0 else (np.nan, np.nan),
        "z": float(z), "p_value": float(2 * st.norm.sf(abs(z))),
    }


def welch_ttest(a, b, alpha: float = 0.05) -> dict:
    """a = control, b = treatment."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    diff = b.mean() - a.mean()
    se = np.sqrt(va + vb)
    df = (va + vb) ** 2 / (va**2 / (len(a) - 1) + vb**2 / (len(b) - 1))
    t = diff / se
    h = st.t.ppf(1 - alpha / 2, df) * se
    return {
        "mean_control": a.mean(), "mean_treatment": b.mean(), "diff": diff, "diff_ci": (diff - h, diff + h),
        "rel_lift": diff / a.mean() if a.mean() else np.nan,
        "rel_lift_ci": ((diff - h) / a.mean(), (diff + h) / a.mean()) if a.mean() else (np.nan, np.nan),
        "t": float(t), "df": float(df), "p_value": float(2 * st.t.sf(abs(t), df)), "se": float(se),
    }


def chi2_independence(df: pd.DataFrame, row: str, col: str) -> dict:
    table = pd.crosstab(df[row], df[col])
    chi2, p, dof, _ = st.chi2_contingency(table)
    n = table.values.sum()
    cramers_v = np.sqrt(chi2 / (n * (min(table.shape) - 1)))
    return {"chi2": float(chi2), "dof": int(dof), "p_value": float(p), "cramers_v": float(cramers_v)}


def mann_whitney(a, b) -> dict:
    u, p = st.mannwhitneyu(a, b, alternative="two-sided")
    # Common-language effect size: P(B > A).
    return {"U": float(u), "p_value": float(p), "prob_superiority": float(u / (len(a) * len(b)))}


def benjamini_hochberg(pvals, q: float = 0.05) -> np.ndarray:
    """Return BH-adjusted p-values (FDR control)."""
    p = np.asarray(pvals, float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    adj = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(adj, 1.0)
    return out


def robust_z(x) -> np.ndarray:
    x = np.asarray(x, float)
    med = np.median(x)
    mad = np.median(np.abs(x - med)) * 1.4826
    return (x - med) / (mad if mad > 0 else 1e-9)
