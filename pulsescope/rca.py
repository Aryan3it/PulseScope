"""Root-cause analysis engine for KPI movements.

1. Detect anomalous days per KPI (STL decomposition with weekly seasonality, robust z on residuals).
2. Group anomalous days into incidents and build a same-weekday baseline.
3. Localise the funnel step that explains the movement (log decomposition of conversion).
4. Drill down across dimensions (Adtributor-style explanatory power + surprise), then one level deeper.
5. Decompose ratio-metric changes into mix vs. rate effects.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL

from . import config as C
from . import sqlrun
from . import stats as S

KPIS = ["sessions", "dau", "session_conversion", "checkout_to_purchase", "revenue", "aov"]
DIMENSIONS = ["platform", "app_version", "geo_region", "city_tier", "traffic_source"]
FUNNEL = [("session_to_cart", "sessions", "carts"), ("cart_to_checkout", "carts", "checkouts"),
          ("checkout_to_purchase", "checkouts", "orders")]
Z_THRESHOLD = 3.5
# Practical-significance floors: a day must be both statistically unusual and move the KPI materially.
MATERIALITY = {"sessions": 0.05, "dau": 0.05, "session_conversion": 0.08, "checkout_to_purchase": 0.08,
               "revenue": 0.12, "aov": 0.12}


# ---------------------------------------------------------------- detection
def detect_anomalies(kpis: pd.DataFrame) -> pd.DataFrame:
    out = kpis[["event_date", "day"]].copy()
    for k in KPIS:
        # A long trend window keeps multi-day incidents in the residual instead of the trend.
        res = STL(kpis[k].to_numpy(float), period=7, trend=29, robust=True).fit()
        out[f"{k}_z"] = S.robust_z(res.resid)
        out[f"{k}_expected"] = res.trend + res.seasonal
        out[f"{k}_rel"] = kpis[k].to_numpy(float) / out[f"{k}_expected"] - 1
        out[f"{k}_flag"] = (out[f"{k}_z"].abs() > Z_THRESHOLD) & (out[f"{k}_rel"].abs() > MATERIALITY[k])
    flags = [f"{k}_flag" for k in KPIS]
    out["is_anomaly"] = out[flags].any(axis=1)
    out["anomalous_kpis"] = out[flags].apply(lambda r: [c[:-5] for c in flags if r[c]], axis=1)
    return out


def group_incidents(anom: pd.DataFrame, max_gap: int = 1) -> list[dict]:
    days = anom.loc[anom.is_anomaly, "day"].tolist()
    groups, cur = [], []
    for d in days:
        if cur and d - cur[-1] > max_gap + 1:
            groups.append(cur)
            cur = []
        cur.append(d)
    if cur:
        groups.append(cur)
    out = []
    for g in groups:
        rows = anom[anom.day.between(g[0], g[-1])]
        kp = sorted({k for ks in rows.anomalous_kpis for k in ks})
        worst = max(kp, key=lambda k: rows[f"{k}_z"].abs().max())
        out.append({"start_day": g[0], "end_day": g[-1], "start": C.date_str(g[0]), "end": C.date_str(g[-1]),
                    "kpis": kp, "primary_kpi": worst,
                    "direction": "down" if rows[f"{worst}_z"].iloc[rows[f"{worst}_z"].abs().argmax()] < 0
                    else "up"})
    return out


def baseline_days(incident: dict, anomalous: set[int], weeks: int = 3) -> list[int]:
    """Same weekdays from the preceding `weeks` weeks, skipping other anomalous days."""
    days = []
    for d in range(incident["start_day"], incident["end_day"] + 1):
        days += [d - 7 * w for w in range(1, weeks + 1) if d - 7 * w >= 0 and d - 7 * w not in anomalous]
    return sorted(set(days))


# ---------------------------------------------------------------- localisation
def funnel_localisation(daily: pd.DataFrame, cur_days, base_days) -> pd.DataFrame:
    cur = daily[daily.day.isin(cur_days)].sum(numeric_only=True)
    base = daily[daily.day.isin(base_days)].sum(numeric_only=True)
    rows = []
    for step, den, num in FUNNEL:
        z = S.two_proportion_ztest(base[num], base[den], cur[num], cur[den])
        rows.append({"step": step, "baseline_rate": z["p_control"], "incident_rate": z["p_treatment"],
                     "rel_change": z["rel_lift"], "log_change": np.log(z["p_treatment"] / z["p_control"]),
                     "p_value": z["p_value"]})
    df = pd.DataFrame(rows)
    df["share_of_conversion_change"] = df.log_change / df.log_change.sum()
    return df


def _js_divergence(p, q) -> np.ndarray:
    """Per-element Jensen-Shannon surprise terms used by Adtributor."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    m = (p + q) / 2
    with np.errstate(divide="ignore", invalid="ignore"):
        a = np.where(p > 0, p * np.log(p / m), 0)
        b = np.where(q > 0, q * np.log(q / m), 0)
    return 0.5 * (a + b)


def drilldown(slices: pd.DataFrame, cur_days, base_days, num: str, den: str | None) -> pd.DataFrame:
    """Explanatory power of each slice for the change in `num` (rate metric if `den` is given).

    For a rate metric the forecast for a slice is baseline_rate * current denominator, i.e. what the
    slice would have produced had its rate not changed. For a volume metric it is the baseline daily mean.
    """
    n_cur, n_base = len(cur_days), len(base_days)
    c = slices[slices.day.isin(cur_days)].groupby("slice").sum(numeric_only=True)
    b = slices[slices.day.isin(base_days)].groupby("slice").sum(numeric_only=True)
    if not den:
        # A rollover (e.g. app 3.8.2 replaced by 3.9.0) substitutes traffic rather than adding it. Pair each
        # vanished slice with the new slice closest to its baseline volume and compare them as one bucket, so a
        # release is not mistaken for a spike; unpaired new slices (e.g. a campaign source) stay additive.
        new = list(c.index.difference(b.index))
        ren = {}
        for g in b.index.difference(c.index):
            if not new:
                break
            vol = b.loc[g, num] / n_base
            n = min(new, key=lambda s: abs(c.loc[s, num] / n_cur - vol) / vol)
            new.remove(n)
            ren[g] = ren[n] = f"{g}→{n}"
        if ren:
            c = c.rename(index=ren).groupby(level=0).sum()
            b = b.rename(index=ren).groupby(level=0).sum()
    df = pd.DataFrame(index=c.index.union(b.index))
    df["actual"] = c[num].reindex(df.index).fillna(0) / n_cur
    if den:
        base_rate = (b[num] / b[den]).reindex(df.index)
        df["denominator"] = c[den].reindex(df.index).fillna(0) / n_cur
        df["baseline_rate"] = base_rate
        df["current_rate"] = df.actual / df.denominator.replace(0, np.nan)
        # Slices with no baseline (e.g. a brand-new app version) inherit the parent's baseline rate.
        parent_rate = b[num].sum() / b[den].sum()
        df["expected"] = (base_rate.fillna(parent_rate) * df.denominator).fillna(0)
        pv = []
        for s in df.index:
            if s in b.index and s in c.index and b.loc[s, den] > 0 and c.loc[s, den] > 0:
                pv.append(S.two_proportion_ztest(b.loc[s, num], b.loc[s, den], c.loc[s, num], c.loc[s, den])
                          ["p_value"])
            else:
                pv.append(np.nan)
        df["p_value"] = pv
    else:
        df["expected"] = b[num].reindex(df.index).fillna(0) / n_base
    total_delta = df.actual.sum() - df.expected.sum()
    df["delta"] = df.actual - df.expected
    df["explanatory_power"] = df.delta / total_delta if total_delta else 0.0
    df["surprise"] = _js_divergence(df.expected / df.expected.sum(), df.actual / df.actual.sum())
    return df.reset_index().rename(columns={"index": "slice"}).sort_values("explanatory_power",
                                                                           ascending=False)


def adtributor(per_dim: dict[str, pd.DataFrame], ep_min: float = 0.10, ep_target: float = 0.67) -> pd.DataFrame:
    """Rank dimensions by surprise of their candidate root-cause slices (Bhagwan et al., NSDI'14)."""
    rows = []
    for dim, df in per_dim.items():
        cand, cum = [], 0.0
        for _, r in df.sort_values("surprise", ascending=False).iterrows():
            if r.explanatory_power > ep_min:
                cand.append(r)
                cum += r.explanatory_power
            if cum > ep_target:
                break
        if cum > ep_target:
            rows.append({"dimension": dim, "slices": [str(r.slice) for r in cand],
                         "explanatory_power": cum, "surprise": float(sum(r.surprise for r in cand))})
    return pd.DataFrame(rows).sort_values("surprise", ascending=False) if rows else pd.DataFrame()


def mix_rate_decomposition(slices: pd.DataFrame, cur_days, base_days, num: str, den: str) -> pd.DataFrame:
    """ΔR = Σ w0·Δr (rate) + Σ Δw·r0 (mix) + Σ Δw·Δr (interaction)."""
    c = slices[slices.day.isin(cur_days)].groupby("slice")[[num, den]].sum()
    b = slices[slices.day.isin(base_days)].groupby("slice")[[num, den]].sum()
    idx = c.index.intersection(b.index)
    c, b = c.loc[idx], b.loc[idx]
    w0, w1 = b[den] / b[den].sum(), c[den] / c[den].sum()
    r0, r1 = b[num] / b[den], c[num] / c[den]
    return pd.DataFrame({"rate_effect": w0 * (r1 - r0), "mix_effect": (w1 - w0) * r0,
                         "interaction": (w1 - w0) * (r1 - r0)})


# ---------------------------------------------------------------- orchestration
def investigate(incident: dict, anomalous: set[int], slice_data: dict[str, pd.DataFrame]) -> dict:
    cur = list(range(incident["start_day"], incident["end_day"] + 1))
    base = baseline_days(incident, anomalous)
    total = slice_data["platform"].groupby("day").sum(numeric_only=True).reset_index()
    funnel = funnel_localisation(total, cur, base)

    # Volume anomalies are explained on sessions; rate anomalies on the funnel step that moved most.
    volume = incident["primary_kpi"] in ("sessions", "dau")
    if volume:
        num, den, metric = "sessions", None, "sessions"
    else:
        step = funnel.loc[funnel.log_change.abs().idxmax()]
        _, den, num = next(f for f in FUNNEL if f[0] == step.step)
        metric = step.step
    per_dim = {d: drilldown(slice_data[d], cur, base, num, den) for d in DIMENSIONS}
    ranking = adtributor(per_dim)
    result = {**incident, "baseline_days": [C.date_str(d) for d in base], "metric_explained": metric,
              "funnel": funnel.to_dict("records"), "dimension_ranking": ranking.to_dict("records"),
              "drilldown": {d: df.head(6).to_dict("records") for d, df in per_dim.items()}}
    if not ranking.empty:
        top_dim = ranking.iloc[0].dimension
        top_slice = ranking.iloc[0].slices[0]
        result["root_cause"] = {"dimension": top_dim, "value": top_slice,
                                "explanatory_power": float(per_dim[top_dim].set_index("slice")
                                                           .loc[top_slice, "explanatory_power"])}
        if den:
            mix = mix_rate_decomposition(slice_data["platform"], cur, base, num, den)
            result["mix_vs_rate"] = {"rate": float(mix.rate_effect.sum()), "mix": float(mix.mix_effect.sum()),
                                     "interaction": float(mix.interaction.sum())}
        # Second level: within the culprit slice, which other dimension concentrates the change?
        # (Skipped for a slice with no baseline traffic, e.g. a brand-new release: nothing to compare against.)
        second = {}
        has_base = top_slice in set(slice_data[top_dim].loc[slice_data[top_dim].day.isin(base), "slice"].astype(str))
        for d in DIMENSIONS if has_base else []:
            if d == top_dim:
                continue
            sl = sqlrun.query(
                "SELECT day, {d} AS slice, COUNT(*) AS sessions, SUM(carted) AS carts, SUM(checkout) AS checkouts,"
                " SUM(purchased) AS orders FROM sessions WHERE {td} = '{tv}' GROUP BY ALL",
                d=d, td=top_dim, tv=top_slice)
            dd = drilldown(sl, cur, base, num, den)
            second[d] = dd.head(3).to_dict("records")
        result["second_level"] = second
    return result


def run() -> dict:
    kpis = sqlrun.query("SELECT * FROM daily_kpis ORDER BY day")
    anom = detect_anomalies(kpis)
    anom.drop(columns="anomalous_kpis").to_csv(C.OUTPUTS / "kpi_anomalies.csv", index=False)
    incidents = group_incidents(anom)
    anomalous = set(anom.loc[anom.is_anomaly, "day"])
    slice_data = {d: sqlrun.run_file("slice_daily", dim=d) for d in DIMENSIONS}
    findings = [investigate(i, anomalous, slice_data) for i in incidents]
    truth = json.loads((C.RAW / "ground_truth.json").read_text())
    out = {"incidents": findings, "evaluation": evaluate(findings, truth)}
    (C.OUTPUTS / "rca.json").write_text(json.dumps(out, indent=2, default=_jd))
    return out


def evaluate(findings: list[dict], truth: dict) -> dict:
    inc = truth["incident"]
    t_days = set(pd.date_range(inc["start"], inc["end"]).strftime("%Y-%m-%d"))
    best = None
    for f in findings:
        f_days = set(pd.date_range(f["start"], f["end"]).strftime("%Y-%m-%d"))
        jac = len(f_days & t_days) / len(f_days | t_days)
        if jac > 0 and (best is None or jac > best[0]):
            best = (jac, f)
    ev = {"incident_detected": best is not None}
    if best:
        jac, f = best
        rc = f.get("root_cause", {})
        second_versions = [r["slice"] for r in f.get("second_level", {}).get("app_version", [])[:1]]
        ev.update({"window_jaccard": jac, "detected_window": [f["start"], f["end"]],
                   "funnel_step_correct": f["metric_explained"] == inc["funnel_step"],
                   "root_cause_found": rc,
                   "root_cause_correct": (rc.get("dimension"), rc.get("value")) in
                   {("app_version", inc["app_version"]), ("platform", inc["platform"])}
                   or second_versions == [inc["app_version"]]})
    camp = truth["campaign"]
    cf = [f for f in findings if f["start"] <= camp["date"] <= f["end"]]
    ev["campaign_detected"] = bool(cf)
    ev["campaign_attributed_correctly"] = bool(cf) and cf[0].get("root_cause", {}).get("value") == camp["value"]
    rh = truth["red_herring"]
    ev["red_herring_false_alarm"] = any(
        rh["app_version"] in str(f.get("root_cause", {}).get("value", "")) for f in findings)
    ev["n_incidents_flagged"] = len(findings)
    ev["false_alarms"] = sum(1 for f in findings
                             if not (f["start"] <= camp["date"] <= f["end"])
                             and not set(pd.date_range(f["start"], f["end"]).strftime("%Y-%m-%d")) & t_days)
    return ev


def _jd(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (pd.Timestamp,)):
        return o.isoformat()
    return str(o)
