"""A/B testing engine: power, SRM, frequentist tests, CUPED, bootstrap, HTE with FDR control,
group-sequential monitoring, and A/A calibration."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import stats as st

from . import config as C
from . import sqlrun
from . import stats as S

ALPHA = 0.05
SRM_ALPHA = 0.001


# ---------------------------------------------------------------- design
def sample_size_proportion(p_base: float, mde_rel: float, alpha=ALPHA, power=0.8) -> int:
    """Users per arm to detect a relative lift `mde_rel` on a conversion rate."""
    p2 = p_base * (1 + mde_rel)
    za, zb = st.norm.ppf(1 - alpha / 2), st.norm.ppf(power)
    pbar = (p_base + p2) / 2
    n = (za * np.sqrt(2 * pbar * (1 - pbar)) + zb * np.sqrt(p_base * (1 - p_base) + p2 * (1 - p2))) ** 2 \
        / (p2 - p_base) ** 2
    return int(np.ceil(n))


def mde_mean(std: float, mean: float, n_per_arm: int, alpha=ALPHA, power=0.8) -> float:
    """Minimum detectable relative effect on a mean metric for a given per-arm sample."""
    se = std * np.sqrt(2 / n_per_arm)
    return (st.norm.ppf(1 - alpha / 2) + st.norm.ppf(power)) * se / mean


# ---------------------------------------------------------------- validity
def srm_test(n_control: int, n_treatment: int, expected_treatment_share: float = 0.5) -> dict:
    total = n_control + n_treatment
    exp = [total * (1 - expected_treatment_share), total * expected_treatment_share]
    chi2, p = st.chisquare([n_control, n_treatment], exp)
    return {"n_control": int(n_control), "n_treatment": int(n_treatment),
            "treatment_share": n_treatment / total, "chi2": float(chi2), "p_value": float(p),
            "srm_detected": bool(p < SRM_ALPHA)}


def srm_by_dimension(df: pd.DataFrame, dim: str) -> pd.DataFrame:
    """Localise a sample-ratio mismatch to the slice that causes it."""
    rows = []
    for v, g in df.groupby(dim):
        r = srm_test((g.variant == "control").sum(), (g.variant == "treatment").sum())
        rows.append({dim: v, **r})
    return pd.DataFrame(rows).sort_values("p_value")


# ---------------------------------------------------------------- estimation
def cuped(df: pd.DataFrame, metric: str, covariate: str) -> dict:
    """CUPED (Deng et al. 2013): y_adj = y - theta * (x - mean(x)), theta pooled across arms."""
    y, x = df[metric].to_numpy(float), df[covariate].to_numpy(float)
    theta = np.cov(y, x)[0, 1] / x.var(ddof=1)
    y_adj = y - theta * (x - x.mean())
    is_t = (df.variant == "treatment").to_numpy()
    raw = S.welch_ttest(y[~is_t], y[is_t])
    adj = S.welch_ttest(y_adj[~is_t], y_adj[is_t])
    return {"theta": float(theta), "corr": float(np.corrcoef(y, x)[0, 1]),
            "variance_reduction": float(1 - y_adj.var() / y.var()),
            "ci_width_reduction": float(1 - (adj["diff_ci"][1] - adj["diff_ci"][0])
                                        / (raw["diff_ci"][1] - raw["diff_ci"][0])),
            "raw": raw, "cuped": adj}


def bootstrap_diff_ci(a, b, n_boot: int = 2000, alpha=ALPHA, seed: int = 0) -> dict:
    """Percentile bootstrap CI for the relative difference in means (robust to heavy-tailed revenue)."""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, float), np.asarray(b, float)
    stats = np.empty(n_boot)
    for i in range(n_boot):
        ma = a[rng.integers(0, len(a), len(a))].mean()
        mb = b[rng.integers(0, len(b), len(b))].mean()
        stats[i] = mb / ma - 1
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"rel_lift": float(b.mean() / a.mean() - 1), "ci": (float(lo), float(hi)), "n_boot": n_boot}


def heterogeneous_effects(df: pd.DataFrame, dims: list[str], metric: str = "converted") -> pd.DataFrame:
    rows = []
    for dim in dims:
        for v, g in df.groupby(dim):
            c, t = g[g.variant == "control"], g[g.variant == "treatment"]
            if len(c) < 200 or len(t) < 200:
                continue
            r = S.two_proportion_ztest(c[metric].sum(), len(c), t[metric].sum(), len(t))
            rows.append({"dimension": dim, "value": v, "n": len(g), "p_control": r["p_control"],
                         "p_treatment": r["p_treatment"], "rel_lift": r["rel_lift"],
                         "rel_lift_lo": r["rel_lift_ci"][0], "rel_lift_hi": r["rel_lift_ci"][1],
                         "p_value": r["p_value"]})
    out = pd.DataFrame(rows)
    out["p_bh"] = S.benjamini_hochberg(out.p_value)
    out["significant_fdr5"] = out.p_bh < ALPHA
    # Interaction test per dimension: is the lift different across slices? (Cochran's Q on log RR)
    q = {}
    for dim, g in out.groupby("dimension"):
        lr = np.log1p(g.rel_lift)
        se = (np.log1p(g.rel_lift_hi) - np.log1p(g.rel_lift_lo)) / (2 * 1.96)
        w = 1 / se**2
        Q = float((w * (lr - (w * lr).sum() / w.sum()) ** 2).sum())
        q[dim] = st.chi2.sf(Q, len(g) - 1)
    out["heterogeneity_p"] = out.dimension.map(q)
    return out


# ---------------------------------------------------------------- sequential
def obf_boundaries(info_fractions, alpha=ALPHA, n_sim: int = 200_000, seed: int = 1) -> np.ndarray:
    """O'Brien-Fleming z boundaries c/sqrt(t_k), with c calibrated by Monte Carlo on Brownian motion
    so the two-sided family-wise type-I error over all looks equals alpha."""
    t = np.asarray(info_fractions, float)
    rng = np.random.default_rng(seed)
    inc = rng.standard_normal((n_sim, len(t))) * np.sqrt(np.diff(np.r_[0, t]))
    z = np.cumsum(inc, axis=1) / np.sqrt(t)
    lo, hi = 1.5, 5.0
    for _ in range(50):
        c = (lo + hi) / 2
        fwer = (np.abs(z) > c / np.sqrt(t)).any(axis=1).mean()
        lo, hi = (c, hi) if fwer > alpha else (lo, c)
    return c / np.sqrt(t)


def peeking_simulation(p: float = 0.05, n_per_look: int = 1000, looks: int = 10, n_sim: int = 4000,
                       seed: int = 2) -> dict:
    """A/A simulation: false-positive rate of naive daily peeking vs. O'Brien-Fleming boundaries."""
    rng = np.random.default_rng(seed)
    xa = rng.binomial(n_per_look, p, (n_sim, looks)).cumsum(1)
    xb = rng.binomial(n_per_look, p, (n_sim, looks)).cumsum(1)
    n = n_per_look * np.arange(1, looks + 1)
    pool = (xa + xb) / (2 * n)
    z = (xb / n - xa / n) / np.sqrt(pool * (1 - pool) * 2 / n)
    bounds = obf_boundaries(np.arange(1, looks + 1) / looks)
    return {"looks": looks, "naive_fpr": float((np.abs(z) > 1.96).any(1).mean()),
            "single_look_fpr": float((np.abs(z[:, -1]) > 1.96).mean()),
            "obf_fpr": float((np.abs(z) > bounds).any(1).mean())}


def sequential_monitor(experiment_id: str, start: int, end: int, step: int = 2) -> pd.DataFrame:
    df = sqlrun.run_file("experiment_sequential", experiment_id=experiment_id, start=start, end=end, step=step)
    w = df.pivot(index="look_day", columns="variant", values=["users", "conversions"])
    rows = []
    for day, r in w.iterrows():
        z = S.two_proportion_ztest(r[("conversions", "control")], r[("users", "control")],
                                   r[("conversions", "treatment")], r[("users", "treatment")])
        rows.append({"look_day": int(day), "date": C.date_str(day),
                     "users": int(r[("users", "control")] + r[("users", "treatment")]),
                     "rel_lift": z["rel_lift"], "z": z["z"], "p_value": z["p_value"]})
    out = pd.DataFrame(rows)
    out["info_fraction"] = out.users / out.users.iloc[-1]
    out["obf_boundary"] = obf_boundaries(out.info_fraction)
    out["crossed_obf"] = out.z.abs() > out.obf_boundary
    out["crossed_naive"] = out.p_value < ALPHA
    return out


def aa_calibration(df: pd.DataFrame, metric: str = "converted", n_sims: int = 1000, seed: int = 3) -> dict:
    """Shuffle labels within the control arm to verify the test's false-positive rate is ~alpha."""
    rng = np.random.default_rng(seed)
    y = df.loc[df.variant == "control", metric].to_numpy()
    pvals = np.empty(n_sims)
    for i in range(n_sims):
        mask = rng.random(len(y)) < 0.5
        pvals[i] = S.two_proportion_ztest(y[~mask].sum(), (~mask).sum(), y[mask].sum(), mask.sum())["p_value"]
    return {"n_sims": n_sims, "false_positive_rate": float((pvals < ALPHA).mean()),
            "ks_uniform_p": float(st.kstest(pvals, "uniform").pvalue)}


# ---------------------------------------------------------------- report
def analyze(experiment_id: str, segments: pd.DataFrame | None = None) -> dict:
    df = sqlrun.query("SELECT * FROM experiment_users WHERE experiment_id = '{e}'", e=experiment_id)
    n_contaminated = int((df.n_variants > 1).sum())
    df = df[df.n_variants == 1].copy()
    if segments is not None:
        df = df.merge(segments[["user_id", "segment"]], on="user_id", how="left")
        df["segment"] = df.segment.fillna("new_or_inactive")
    df["carted"] = (df.carts > 0).astype(int)

    srm = srm_test((df.variant == "control").sum(), (df.variant == "treatment").sum())
    res = {"experiment_id": experiment_id, "n_users": len(df), "contaminated_users_removed": n_contaminated,
           "srm": srm}
    if srm["srm_detected"]:
        res["srm_localisation"] = srm_by_dimension(df, "platform").to_dict("records")

    c, t = df[df.variant == "control"], df[df.variant == "treatment"]
    res["conversion"] = S.two_proportion_ztest(c.converted.sum(), len(c), t.converted.sum(), len(t))
    res["cart_rate"] = S.two_proportion_ztest(c.carted.sum(), len(c), t.carted.sum(), len(t))
    res["revenue_per_user"] = cuped(df, "revenue", "pre_revenue")
    res["orders_per_user"] = cuped(df, "orders", "pre_orders")
    res["revenue_bootstrap"] = bootstrap_diff_ci(c.revenue, t.revenue)
    res["guardrail_sessions"] = S.welch_ttest(c.sessions, t.sessions)
    res["revenue_mann_whitney"] = S.mann_whitney(c.revenue, t.revenue)
    p0 = c.converted.mean()
    res["design"] = {"baseline_conversion": p0,
                     "n_per_arm_for_3pct_mde": sample_size_proportion(p0, 0.03),
                     "n_per_arm_for_5pct_mde": sample_size_proportion(p0, 0.05),
                     "revenue_mde_rel_at_observed_n": mde_mean(c.revenue.std(), c.revenue.mean(), len(c))}
    dims = ["platform", "city_tier"] + (["segment"] if segments is not None else [])
    hte = heterogeneous_effects(df, dims)
    res["hte"] = hte.to_dict("records")

    conv = res["conversion"]
    guard_ok = res["guardrail_sessions"]["p_value"] > ALPHA or res["guardrail_sessions"]["diff"] > 0
    if srm["srm_detected"]:
        decision = "INVALID - sample ratio mismatch; fix exposure logging and rerun before reading results"
    elif conv["p_value"] < ALPHA and conv["diff"] > 0 and guard_ok:
        decision = "SHIP - significant lift on primary metric, guardrails healthy"
    elif conv["p_value"] < ALPHA and conv["diff"] < 0:
        decision = "DO NOT SHIP - significant regression on primary metric"
    else:
        decision = "INCONCLUSIVE - no significant effect at current power"
    res["decision"] = decision
    df.to_parquet(C.OUTPUTS / f"exp_{experiment_id}_users.parquet", index=False)
    df.drop(columns=[c for c in ["segment"] if c in df]).to_csv(C.OUTPUTS / f"exp_{experiment_id}_users.csv",
                                                                index=False)
    return res


def run(segments: pd.DataFrame | None = None) -> dict:
    out = {}
    for exp in (C.EXP_A, C.EXP_B):
        r = analyze(exp["id"], segments)
        seq = sequential_monitor(exp["id"], exp["start"], exp["end"])
        seq.to_csv(C.OUTPUTS / f"exp_{exp['id']}_sequential.csv", index=False)
        r["sequential"] = {"first_obf_crossing": seq.loc[seq.crossed_obf, "date"].min()
                           if seq.crossed_obf.any() else None,
                           "first_naive_crossing": seq.loc[seq.crossed_naive, "date"].min()
                           if seq.crossed_naive.any() else None,
                           "looks": len(seq)}
        out[exp["id"]] = r
    df_a = pd.read_parquet(C.OUTPUTS / f"exp_{C.EXP_A['id']}_users.parquet")
    out["aa_calibration"] = aa_calibration(df_a)
    out["peeking_simulation"] = peeking_simulation()
    (C.OUTPUTS / "experiments.json").write_text(json.dumps(out, indent=2, default=_json_default))
    return out


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (pd.Timestamp,)):
        return o.isoformat()
    return str(o)
