"""Builds outputs/report.html: a self-contained insight report (Plotly charts + findings + recommendations)."""
from __future__ import annotations

import html
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from . import config as C
from . import sqlrun

# Reference data-viz palette (validated categorical order; ink/grid tokens; status reserved).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
CRITICAL, GOOD = "#d03b3b", "#0ca30c"
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"


def _fig(fig: go.Figure, title: str, height: int = 340, yfmt: str | None = None) -> str:
    fig.update_layout(
        title=dict(text=title, x=0, font=dict(size=15, color=INK)),
        height=height, margin=dict(l=56, r=16, t=48, b=72 if len(fig.data) > 1 else 44),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif", size=12, color=INK2),
        legend=dict(orientation="h", y=-0.14, yanchor="top", x=0, xanchor="left", title=None),
        hoverlabel=dict(bgcolor="#ffffff", font_color=INK), hovermode="closest",
    )
    fig.update_xaxes(gridcolor=GRID, linecolor=AXIS, zeroline=False, tickfont_color=MUTED)
    fig.update_yaxes(gridcolor=GRID, linecolor=AXIS, zeroline=False, tickfont_color=MUTED)
    if yfmt:
        fig.update_yaxes(tickformat=yfmt)
    return fig.to_html(full_html=False, include_plotlyjs=False, config={"displayModeBar": False,
                                                                         "responsive": True})


def _table(df: pd.DataFrame, fmt: dict | None = None) -> str:
    fmt = fmt or {}
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns)
    rows = []
    for _, r in df.iterrows():
        cells = []
        for c in df.columns:
            v = r[c]
            if c in fmt and pd.notna(v):
                v = fmt[c](v)
            elif isinstance(v, float):
                v = f"{v:,.3f}"
            cells.append(f"<td>{html.escape(str(v))}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return f"<div class='tbl'><table><thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"


pct = lambda v: f"{v:.1%}"          # noqa: E731
pct2 = lambda v: f"{v:+.2%}"        # noqa: E731
num = lambda v: f"{v:,.0f}"         # noqa: E731
pval = lambda v: "<0.0001" if v < 1e-4 else f"{v:.4f}"  # noqa: E731


def _load(name):
    return json.loads((C.OUTPUTS / name).read_text())


def kpi_section(rca: dict) -> str:
    k = sqlrun.query("SELECT * FROM daily_kpis ORDER BY day")
    an = pd.read_csv(C.OUTPUTS / "kpi_anomalies.csv")
    k = k.merge(an[["day", "is_anomaly", "sessions_expected", "checkout_to_purchase_expected"]], on="day")
    out = []
    for metric, label, fmt in [("sessions", "Daily sessions", ",.0f"),
                               ("checkout_to_purchase", "Checkout → purchase rate", ".0%")]:
        fig = go.Figure()
        fig.add_scatter(x=k.event_date, y=k[f"{metric}_expected"], name="Expected",
                        line=dict(color=MUTED, width=2, dash="dot"))
        fig.add_scatter(x=k.event_date, y=k[metric], name="Actual", line=dict(color=SERIES[0], width=2))
        a = k[k.is_anomaly]
        fig.add_scatter(x=a.event_date, y=a[metric], name="Anomaly", mode="markers",
                        marker=dict(color=CRITICAL, size=9, symbol="diamond", line=dict(color="#fcfcfb", width=2)))
        for inc in rca["incidents"]:
            fig.add_vrect(x0=inc["start"], x1=pd.Timestamp(inc["end"]) + pd.Timedelta(days=1),
                          fillcolor=CRITICAL, opacity=0.07, line_width=0)
        out.append(_fig(fig, label, yfmt=fmt))
    return "".join(out)


def rca_section(rca: dict) -> str:
    parts = []
    ev = rca["evaluation"]
    for inc in rca["incidents"]:
        rc = inc.get("root_cause", {})
        f = pd.DataFrame(inc["funnel"])
        parts.append(f"<h3>{inc['start']} → {inc['end']}: {inc['primary_kpi']} {inc['direction']}</h3>")
        parts.append(f"<p>Explained metric: <b>{inc['metric_explained']}</b>. "
                     f"Root cause: <b>{rc.get('dimension', '–')} = {rc.get('value', '–')}</b> "
                     f"(explanatory power {rc.get('explanatory_power', float('nan')):.0%}"
                     + ("; above 100% because the other slices moved slightly the other way"
                        if rc.get("explanatory_power", 0) > 1 else "") + ").</p>")
        fig = go.Figure(go.Bar(x=f.step, y=f.rel_change, marker_color=SERIES[0],
                               text=[pct2(v) for v in f.rel_change], textposition="outside", cliponaxis=False,
                               hovertemplate="%{x}<br>change %{y:.2%}<extra></extra>"))
        fig.update_yaxes(range=[min(0, f.rel_change.min()) * 1.3 - 0.002, max(0, f.rel_change.max()) * 1.3 + 0.002])
        funnel_html = _fig(fig, "Funnel step change vs. same-weekday baseline", 280, ".0%")
        dim = rc.get("dimension")
        dd_html = ""
        if dim:
            dd = pd.DataFrame(inc["drilldown"][dim])
            fig = go.Figure(go.Bar(y=dd.slice.astype(str), x=dd.explanatory_power, orientation="h",
                                   marker_color=[SERIES[1] if str(s) == str(rc["value"]) else SERIES[0]
                                                 for s in dd.slice],
                                   hovertemplate="%{y}<br>explanatory power %{x:.0%}<extra></extra>"))
            fig.update_yaxes(autorange="reversed")
            dd_html = _fig(fig, f"Share of the change explained, by {dim}", 280, None)
        parts.append(f"<div class='grid2'>{funnel_html}{dd_html}</div>")
        if inc.get("dimension_ranking"):
            r = pd.DataFrame(inc["dimension_ranking"])
            r["slices"] = r.slices.apply(", ".join)
            parts.append(_table(r, {"explanatory_power": pct}))
    score = pd.DataFrame([{"check": k, "result": v} for k, v in ev.items()])
    return "".join(parts) + "<h3>Scored against planted ground truth</h3>" + _table(score)


def segment_section(seg: dict) -> str:
    prof = pd.read_csv(C.OUTPUTS / "segment_profiles.csv")
    fig = go.Figure()
    fig.add_bar(y=prof.segment, x=prof.user_share, name="Share of users", orientation="h",
                marker_color=SERIES[0], hovertemplate="%{y}<br>users %{x:.1%}<extra></extra>")
    fig.add_bar(y=prof.segment, x=prof.revenue_share, name="Share of revenue", orientation="h",
                marker_color=SERIES[1], hovertemplate="%{y}<br>revenue %{x:.1%}<extra></extra>")
    fig.update_layout(barmode="group", bargap=0.3, bargroupgap=0.08)
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(tickformat=".0%")
    bars = _fig(fig, "Users vs. revenue by segment", 320, None)

    ct = pd.read_csv(C.OUTPUTS / "segment_vs_persona.csv", index_col=0)
    heat = go.Figure(go.Heatmap(z=ct.values, x=ct.columns, y=ct.index, colorscale=SEQ, zmin=0, zmax=1,
                                text=np.vectorize(lambda v: f"{v:.0%}")(ct.values), texttemplate="%{text}",
                                hovertemplate="segment %{y}<br>true persona %{x}<br>%{z:.1%}<extra></extra>",
                                colorbar=dict(tickformat=".0%")))
    heat_html = _fig(heat, "Discovered segment vs. hidden persona (row %)", 320)

    sel = pd.read_csv(C.OUTPUTS / "segmentation_model_selection.csv")
    fig = go.Figure(go.Scatter(x=sel.k, y=sel.inertia, mode="lines+markers", line=dict(color=SERIES[0], width=2),
                               marker=dict(size=8), name="Inertia"))
    fig.add_vline(x=seg["k"], line=dict(color=MUTED, dash="dot"))
    elbow = _fig(fig, f"K-Means inertia (elbow → k = {seg['k']})", 260)
    sil = go.Figure(go.Scatter(x=sel.k, y=sel.silhouette, mode="lines+markers",
                               line=dict(color=SERIES[0], width=2), marker=dict(size=8), name="Silhouette"))
    sil_html = _fig(sil, "Silhouette by k", 260)

    show = prof[["segment", "users", "sessions", "orders", "revenue", "aov", "discount_share", "app_share",
                 "churn_rate", "user_share", "revenue_share"]]
    tbl = _table(show, {"users": num, "revenue": num, "aov": num, "discount_share": pct, "app_share": pct,
                        "churn_rate": pct, "user_share": pct, "revenue_share": pct,
                        "sessions": lambda v: f"{v:.1f}", "orders": lambda v: f"{v:.2f}"})
    m = (f"<p>k = {seg['k']} chosen by the inertia elbow (no access to labels). "
         f"ARI vs. hidden personas <b>{seg['ari_vs_truth']:.3f}</b> (NMI {seg['nmi_vs_truth']:.3f}) vs. "
         f"RFM-only baseline ARI {seg['ari_rfm_baseline']:.3f}. Bootstrap stability ARI "
         f"{seg['bootstrap_stability_ari']:.3f}. Spark MLlib vs scikit-learn agreement ARI "
         f"{seg.get('spark_vs_sklearn_ari', float('nan')):.3f}. A depth-4 surrogate tree reproduces segment "
         f"membership with {seg['surrogate_tree_accuracy']:.1%} accuracy.</p>")
    return m + f"<div class='grid2'>{bars}{heat_html}</div><div class='grid2'>{elbow}{sil_html}</div>" + tbl


def churn_section(ch: dict) -> str:
    rows = [{"model": f"H2O AutoML ({ch['h2o_automl'].get('leader', 'n/a')})", **{
        k: ch["h2o_automl"].get(k) for k in ("auc", "aucpr", "brier", "lift_top_decile")}}]
    rows += [{"model": k, **v} for k, v in ch["baselines"].items()]
    tbl = _table(pd.DataFrame(rows), {"auc": lambda v: f"{v:.3f}", "aucpr": lambda v: f"{v:.3f}",
                                      "brier": lambda v: f"{v:.4f}", "lift_top_decile": lambda v: f"{v:.2f}×"})
    extra = ""
    fi = C.OUTPUTS / "churn_feature_importance.csv"
    if fi.exists():
        v = pd.read_csv(fi).head(10)
        fig = go.Figure(go.Bar(y=v.variable, x=v.scaled_importance, orientation="h", marker_color=SERIES[0],
                               hovertemplate="%{y}<br>scaled importance %{x:.2f}<extra></extra>"))
        fig.update_yaxes(autorange="reversed")
        extra += _fig(fig, "Top churn drivers (best H2O GBM, scaled importance)", 320)
    rk = C.OUTPUTS / "churn_risk_by_segment.csv"
    if rk.exists():
        r = pd.read_csv(rk)
        fig = go.Figure(go.Bar(x=r.segment, y=r.revenue_at_risk_per_100_users, marker_color=SERIES[0],
                               hovertemplate="%{x}<br>₹%{y:,.0f} at risk per 100 users<extra></extra>"))
        extra += _fig(fig, "Expected revenue at risk per 100 users (₹)", 320, ",.0f")
    return (f"<p>Churn = no session in the 31 days after the observation window. Base rate "
            f"{ch['churn_rate']:.1%}; {ch['n_train']:,} train / {ch['n_test']:,} test users.</p>" + tbl
            + f"<div class='grid2'>{extra}</div>")


def experiment_section(ex: dict, rv: pd.DataFrame | None) -> str:
    parts = []
    for eid in (C.EXP_A["id"], C.EXP_B["id"]):
        e = ex[eid]
        conv, rev = e["conversion"], e["revenue_per_user"]
        status = "critical" if e["srm"]["srm_detected"] else ("good" if e["decision"].startswith("SHIP") else "warn")
        parts.append(f"<h3>{eid}</h3><p class='decision {status}'>{html.escape(e['decision'])}</p>")
        summary = pd.DataFrame([
            {"metric": "Users (ctrl / treat)", "value": f"{e['srm']['n_control']:,} / {e['srm']['n_treatment']:,}",
             "detail": f"SRM χ² = {e['srm']['chi2']:.1f}, p = {pval(e['srm']['p_value'])}"},
            {"metric": "Conversion", "value": f"{conv['p_control']:.2%} → {conv['p_treatment']:.2%}",
             "detail": f"lift {conv['rel_lift']:+.2%} [{conv['rel_lift_ci'][0]:+.2%}, {conv['rel_lift_ci'][1]:+.2%}],"
                       f" p = {pval(conv['p_value'])}"},
            {"metric": "Revenue / user (raw)", "value": f"₹{rev['raw']['diff']:+.1f}",
             "detail": f"95% CI [{rev['raw']['diff_ci'][0]:+.1f}, {rev['raw']['diff_ci'][1]:+.1f}],"
                       f" p = {pval(rev['raw']['p_value'])}"},
            {"metric": "Revenue / user (CUPED)", "value": f"₹{rev['cuped']['diff']:+.1f}",
             "detail": f"95% CI [{rev['cuped']['diff_ci'][0]:+.1f}, {rev['cuped']['diff_ci'][1]:+.1f}], "
                       f"p = {pval(rev['cuped']['p_value'])}; variance −{rev['variance_reduction']:.1%}"},
            {"metric": "Revenue lift (bootstrap)", "value": f"{e['revenue_bootstrap']['rel_lift']:+.2%}",
             "detail": f"95% CI [{e['revenue_bootstrap']['ci'][0]:+.2%}, {e['revenue_bootstrap']['ci'][1]:+.2%}]"},
            {"metric": "Guardrail: sessions / user", "value": f"{e['guardrail_sessions']['rel_lift']:+.2%}",
             "detail": f"p = {pval(e['guardrail_sessions']['p_value'])}"},
            {"metric": "Sequential (O'Brien-Fleming)",
             "value": e["sequential"]["first_obf_crossing"] or "boundary never crossed",
             "detail": f"first naive p<0.05: {e['sequential']['first_naive_crossing'] or 'never'}"},
        ])
        parts.append(_table(summary))
        if e["srm"]["srm_detected"] and "srm_localisation" in e:
            loc = pd.DataFrame(e["srm_localisation"])[["platform", "n_control", "n_treatment", "treatment_share",
                                                        "p_value"]]
            parts.append("<p>SRM localised by platform:</p>" + _table(loc, {"treatment_share": pct, "p_value": pval}))
            continue
        h = pd.DataFrame(e["hte"])
        h["label"] = h.dimension + " = " + h.value.astype(str)
        fig = go.Figure(go.Scatter(
            x=h.rel_lift, y=h.label, mode="markers",
            error_x=dict(type="data", symmetric=False, array=h.rel_lift_hi - h.rel_lift,
                         arrayminus=h.rel_lift - h.rel_lift_lo, color=MUTED, thickness=2),
            marker=dict(size=9, color=[SERIES[1] if s else SERIES[0] for s in h.significant_fdr5]),
            customdata=np.c_[h.p_bh, h.n],
            hovertemplate="%{y}<br>lift %{x:.2%}<br>BH-adjusted p %{customdata[0]:.4f}<br>n %{customdata[1]:,}"
                          "<extra></extra>"))
        fig.add_vline(x=0, line=dict(color=AXIS))
        fig.add_vline(x=conv["rel_lift"], line=dict(color=MUTED, dash="dot"))
        fig.update_yaxes(autorange="reversed")
        fig.update_xaxes(tickformat=".0%")
        forest = _fig(fig, "Conversion lift by slice (95% CI; orange = significant after BH-FDR)",
                      90 + 26 * len(h))
        seq = pd.read_csv(C.OUTPUTS / f"exp_{eid}_sequential.csv")
        fig = go.Figure()
        fig.add_scatter(x=seq.date, y=seq.obf_boundary, name="O'Brien-Fleming boundary",
                        line=dict(color=MUTED, width=2, dash="dot"))
        fig.add_scatter(x=seq.date, y=np.full(len(seq), 1.96), name="Naive z = 1.96",
                        line=dict(color=AXIS, width=2, dash="dash"))
        fig.add_scatter(x=seq.date, y=seq.z, name="Observed z", line=dict(color=SERIES[0], width=2),
                        mode="lines+markers", marker=dict(size=8))
        seq_html = _fig(fig, "Interim looks: cumulative z vs. stopping boundary", 300)
        parts.append(f"<div class='grid2'>{forest}{seq_html}</div>")
    aa, pk = ex["aa_calibration"], ex["peeking_simulation"]
    parts.append(f"<h3>Is the testing machinery trustworthy?</h3><p>A/A calibration on real control-arm data: "
                 f"false-positive rate <b>{aa['false_positive_rate']:.1%}</b> over {aa['n_sims']:,} shuffles "
                 f"(p-value uniformity KS p = {aa['ks_uniform_p']:.2f}). Peeking simulation with {pk['looks']} looks: "
                 f"naive FPR <b>{pk['naive_fpr']:.1%}</b> vs. O'Brien-Fleming <b>{pk['obf_fpr']:.1%}</b>.</p>")
    if rv is not None:
        parts.append("<p>Independent re-analysis in R (prop.test, Welch t.test, ANCOVA, boot):</p>"
                     + _table(rv, {"python": lambda v: f"{v:.6g}", "r": lambda v: f"{v:.6g}",
                                   "abs_diff": lambda v: f"{v:.2e}", "rel_diff": lambda v: f"{v:.2e}"}))
    return "".join(parts)


def sql_section() -> str:
    ret = pd.read_csv(C.OUTPUTS / "sql_retention_cohorts.csv")
    piv = ret.pivot(index="cohort_week", columns="week_n", values="retention").iloc[-10:]
    heat = go.Figure(go.Heatmap(z=piv.values, x=[f"W{c}" for c in piv.columns], y=[str(i)[:10] for i in piv.index],
                                colorscale=SEQ, zmin=0, zmax=1, colorbar=dict(tickformat=".0%"),
                                hovertemplate="cohort %{y}<br>%{x}: %{z:.1%}<extra></extra>"))
    heat.update_yaxes(autorange="reversed")
    fun = pd.read_csv(C.OUTPUTS / "sql_funnel.csv")
    fig = go.Figure()
    for i, (p, g) in enumerate(fun.groupby("platform")):
        fig.add_bar(x=g.stage, y=g.cumulative_conversion, name=p, marker_color=SERIES[i],
                    hovertemplate=f"{p}<br>%{{x}}: %{{y:.1%}} of sessions<extra></extra>")
    fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.08)
    fig.update_yaxes(tickformat=".0%", title="share of sessions")
    eng = pd.read_csv(C.OUTPUTS / "sql_engagement.csv")
    st = go.Figure(go.Scatter(x=eng.event_date, y=eng.stickiness, line=dict(color=SERIES[0], width=2),
                              name="DAU/MAU", hovertemplate="%{x}<br>DAU/MAU %{y:.1%}<extra></extra>"))
    return (f"<div class='grid2'>{_fig(fig, 'Session funnel by platform (cumulative)', 320)}"
            f"{_fig(heat, 'Weekly cohort retention', 320)}</div>"
            f"<div class='grid2'>{_fig(st, 'Stickiness (DAU / rolling-30-day MAU)', 260, '.0%')}</div>")


def recommendations(rca, seg, ch, ex) -> str:
    recs = []
    for inc in rca["incidents"]:
        rc = inc.get("root_cause")
        if not rc:
            continue
        if inc["direction"] == "down":
            sl = inc.get("second_level", {}).get("app_version", [])
            ver = f" (concentrated in app_version {sl[0]['slice']})" if sl and rc["dimension"] != "app_version" else ""
            share = "essentially all" if rc["explanatory_power"] > 0.95 else f"{rc['explanatory_power']:.0%}"
            recs.append(f"<b>Incident {inc['start']}→{inc['end']}</b>: {inc['metric_explained']} fell; "
                        f"{rc['dimension']} = {rc['value']}{ver} explains {share} of the loss. "
                        f"Add a release-gated canary alert on checkout→purchase by app_version.")
        else:
            recs.append(f"<b>Spike {inc['start']}</b>: driven by {rc['dimension']} = {rc['value']} "
                        f"({rc['explanatory_power']:.0%} of excess sessions) - a campaign, not a data issue.")
    a = ex[C.EXP_A["id"]]
    sig = [h for h in a["hte"] if h["significant_fdr5"] and h["dimension"] == "segment"]
    tgt = max(sig, key=lambda h: h["rel_lift"]) if sig else None
    recs.append(f"<b>{C.EXP_A['id']}</b>: {a['decision']}. "
                + (f"Largest segment effect: {tgt['value']} ({tgt['rel_lift']:+.1%}); lead the rollout comms there."
                   if tgt else ""))
    recs.append(f"<b>{C.EXP_B['id']}</b>: {ex[C.EXP_B['id']]['decision']}.")
    rk = C.OUTPUTS / "churn_risk_by_segment.csv"
    if rk.exists():
        r = pd.read_csv(rk).iloc[0]
        recs.append(f"<b>Retention</b>: prioritise <i>{r.segment}</i> for win-back - highest expected revenue at "
                    f"risk (₹{r.revenue_at_risk_per_100_users:,.0f} per 100 users, predicted churn "
                    f"{r.predicted_churn:.0%}).")
    return "<ul>" + "".join(f"<li>{r}</li>" for r in recs) + "</ul>"


CSS = """
:root{--bg:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--grid:#e1e0d9;
--border:rgba(11,11,11,.10);--good:#006300;--crit:#d03b3b;--warn:#8a5a00}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;
--ink2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--border:rgba(255,255,255,.10);--good:#0ca30c;--crit:#e66767;--warn:#fab219}}
:root[data-theme="dark"]{--bg:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--grid:#2c2c2a;
--border:rgba(255,255,255,.10);--good:#0ca30c;--crit:#e66767;--warn:#fab219}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1180px;margin:0 auto;padding:32px 16px 64px}
h1{font-size:28px;margin:0 0 4px}h2{font-size:20px;margin:40px 0 12px;padding-top:16px;border-top:1px solid var(--grid)}
h3{font-size:16px;margin:24px 0 8px}p,li{color:var(--ink2);max-width:80ch}.sub{color:var(--muted)}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin:20px 0}
.tile{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:14px}
.tile .v{font-size:24px;font-weight:600;color:var(--ink)}.tile .l{font-size:12px;color:var(--muted)}
.grid2{display:grid;grid-template-columns:repeat(auto-fill,minmax(min(100%,520px),1fr));gap:12px;margin-bottom:12px}
.grid2>div{min-width:0}
.grid2>div,.plotly-graph-div{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:4px}
.tbl{overflow-x:auto;margin:8px 0 16px}table{border-collapse:collapse;font-size:13px;width:100%}
th,td{text-align:left;padding:6px 10px;border-bottom:1px solid var(--grid);font-variant-numeric:tabular-nums}
th{color:var(--muted);font-weight:600}td{color:var(--ink)}
.decision{font-weight:600;padding:8px 12px;border-radius:8px;border:1px solid var(--border);background:var(--surface)}
.decision.good{color:var(--good)}.decision.good::before{content:"✓ "}
.decision.critical{color:var(--crit)}.decision.critical::before{content:"⚠ "}
.decision.warn{color:var(--warn)}.decision.warn::before{content:"◐ "}
"""

THEME_JS = """
<script>
function themeCharts(){
  const s=getComputedStyle(document.documentElement);const ink=s.getPropertyValue('--ink2').trim(),
  grid=s.getPropertyValue('--grid').trim(),title=s.getPropertyValue('--ink').trim(),surf=s.getPropertyValue('--surface').trim();
  document.querySelectorAll('.plotly-graph-div').forEach(g=>{try{Plotly.relayout(g,{'font.color':ink,'title.font.color':title,
  'xaxis.gridcolor':grid,'yaxis.gridcolor':grid,'hoverlabel.bgcolor':surf,'hoverlabel.font.color':title})}catch(e){}});
}
// Charts are drawn while the page is still laying out; size them to their final cards once loaded.
window.addEventListener('load',()=>{document.querySelectorAll('.plotly-graph-div').forEach(g=>Plotly.Plots.resize(g));themeCharts();});
window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change',themeCharts);
</script>"""


def build() -> str:
    dq, rca, seg, ex = (_load(n) for n in ("data_quality.json", "rca.json", "segmentation.json", "experiments.json"))
    ch = _load("churn.json") if (C.OUTPUTS / "churn.json").exists() else None
    rv = pd.read_csv(C.OUTPUTS / "python_vs_r.csv") if (C.OUTPUTS / "python_vs_r.csv").exists() else None
    truth = json.loads((C.RAW / "ground_truth.json").read_text())
    a = ex[C.EXP_A["id"]]
    rc = next((i.get("root_cause") for i in rca["incidents"] if i["direction"] == "down"), None) or {}
    tiles = [
        (f"{dq['raw_rows']:,}", "raw events processed (Spark)"),
        (f"{truth['n_users']:,}", "users simulated"),
        (f"{rc.get('dimension', '–')} = {rc.get('value', '–')}", "incident root cause"),
        (f"{a['conversion']['rel_lift']:+.1%}", "one-click checkout conversion lift"),
        (f"−{a['revenue_per_user']['variance_reduction']:.0%}", "variance with CUPED"),
        (f"{seg['ari_vs_truth']:.2f}", "segmentation ARI vs. truth"),
    ]
    if ch and "auc" in ch["h2o_automl"]:
        tiles.append((f"{ch['h2o_automl']['auc']:.3f}", "churn AUC (H2O AutoML)"))
    tiles_html = "".join(f"<div class='tile'><div class='v'>{html.escape(v)}</div><div class='l'>{l}</div></div>"
                         for v, l in tiles)
    dqt = pd.DataFrame([{"check": k, "rows": v} for k, v in dq.items() if not isinstance(v, dict)])
    body = f"""
<main>
<h1>PulseScope · ShopKart product health report</h1>
<p class="sub">{C.date_str(0)} → {C.date_str(C.N_DAYS - 1)} · generated from the Parquet lake by <code>pulsescope.report</code></p>
<div class="tiles">{tiles_html}</div>
<h2>Recommendations</h2>{recommendations(rca, seg, ch, ex)}
<h2>1 · KPI monitoring &amp; anomaly detection</h2>
<p>STL decomposition (weekly period, robust) per KPI. A day is flagged only when it is both statistically unusual
(|robust z| &gt; 3.5 on the residual) and material (±5% traffic, ±8% conversion, ±12% revenue/AOV); flagged days
are grouped into incidents (shaded). The z-test alone raised 7 false alarms on sub-3% traffic wiggles.</p>
<div class="grid2">{kpi_section(rca)}</div>
<h2>2 · Root-cause analysis</h2>{rca_section(rca)}
<h2>3 · Funnel, retention &amp; engagement (SQL)</h2>{sql_section()}
<h2>4 · User segmentation</h2>{segment_section(seg)}
<h2>5 · Churn propensity</h2>{churn_section(ch) if ch else '<p>not run</p>'}
<h2>6 · Experiments</h2>{experiment_section(ex, rv)}
<h2>7 · Data quality (Spark ETL)</h2>{_table(dqt, {'rows': lambda v: f'{v:,.0f}' if float(v).is_integer() else f'{v:,.1f}'})}
</main>"""
    page = (f"<!doctype html><html lang='en'><head><meta charset='utf-8'>"
            f"<meta name='viewport' content='width=device-width,initial-scale=1'><title>PulseScope Report</title>"
            f"<style>{CSS}</style>"
            f"<script src='https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js'></script>"
            f"</head><body>{body}{THEME_JS}</body></html>")
    path = C.OUTPUTS / "report.html"
    path.write_text(page)
    print(f"report -> {path}")
    return str(path)
