"""Interactive dashboard over the pipeline outputs: `streamlit run dashboard/app.py`."""
import json
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pulsescope import config as C  # noqa: E402
from pulsescope import experiments, sqlrun  # noqa: E402
from pulsescope import stats as S  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
CRITICAL = "#d03b3b"

st.set_page_config(page_title="PulseScope", layout="wide")
st.title("PulseScope · product analytics")


@st.cache_data
def load_json(name):
    return json.loads((C.OUTPUTS / name).read_text())


@st.cache_data
def q(sql):
    return sqlrun.query(sql)


tab_kpi, tab_rca, tab_seg, tab_exp, tab_calc = st.tabs(
    ["KPIs", "Root cause", "Segments", "Experiments", "Experiment calculator"])

with tab_kpi:
    kpis = q("SELECT * FROM daily_kpis ORDER BY day")
    an = pd.read_csv(C.OUTPUTS / "kpi_anomalies.csv")
    metric = st.selectbox("KPI", ["sessions", "dau", "session_conversion", "checkout_to_purchase", "revenue", "aov"])
    df = kpis.merge(an[["day", "is_anomaly", f"{metric}_expected"]], on="day")
    fig = go.Figure()
    fig.add_scatter(x=df.event_date, y=df[f"{metric}_expected"], name="Expected", line=dict(color="#898781",
                                                                                           dash="dot", width=2))
    fig.add_scatter(x=df.event_date, y=df[metric], name="Actual", line=dict(color=SERIES[0], width=2))
    a = df[df.is_anomaly]
    fig.add_scatter(x=a.event_date, y=a[metric], mode="markers", name="Anomaly",
                    marker=dict(color=CRITICAL, size=9, symbol="diamond"))
    st.plotly_chart(fig, width="stretch")
    c1, c2 = st.columns(2)
    c1.dataframe(pd.read_csv(C.OUTPUTS / "sql_funnel.csv"), hide_index=True)
    c2.dataframe(load_json("data_quality.json")["tables"], width="stretch")

with tab_rca:
    rca = load_json("rca.json")
    st.json(rca["evaluation"], expanded=True)
    for inc in rca["incidents"]:
        with st.expander(f"{inc['start']} → {inc['end']} · {inc['primary_kpi']} {inc['direction']}", expanded=True):
            st.write(f"Root cause: **{inc.get('root_cause')}**")
            st.dataframe(pd.DataFrame(inc["funnel"]), hide_index=True)
            dim = st.selectbox("Drill-down dimension", list(inc["drilldown"]), key=inc["start"])
            st.dataframe(pd.DataFrame(inc["drilldown"][dim]), hide_index=True)

with tab_seg:
    prof = pd.read_csv(C.OUTPUTS / "segment_profiles.csv")
    st.dataframe(prof, hide_index=True)
    seg = load_json("segmentation.json")
    c = st.columns(4)
    c[0].metric("k", seg["k"])
    c[1].metric("ARI vs truth", f"{seg['ari_vs_truth']:.3f}")
    c[2].metric("RFM baseline ARI", f"{seg['ari_rfm_baseline']:.3f}")
    c[3].metric("Stability ARI", f"{seg['bootstrap_stability_ari']:.3f}")
    st.code((C.OUTPUTS / "segment_rules.txt").read_text())

with tab_exp:
    ex = load_json("experiments.json")
    eid = st.radio("Experiment", [C.EXP_A["id"], C.EXP_B["id"]], horizontal=True)
    e = ex[eid]
    (st.error if e["srm"]["srm_detected"] else st.success)(e["decision"])
    c = st.columns(3)
    c[0].metric("Conversion lift", f"{e['conversion']['rel_lift']:+.2%}", f"p={e['conversion']['p_value']:.4f}")
    c[1].metric("Revenue/user (CUPED)", f"₹{e['revenue_per_user']['cuped']['diff']:+.1f}",
                f"p={e['revenue_per_user']['cuped']['p_value']:.4f}")
    c[2].metric("CUPED variance reduction", f"{e['revenue_per_user']['variance_reduction']:.1%}")
    st.dataframe(pd.DataFrame(e["hte"]), hide_index=True)
    st.line_chart(pd.read_csv(C.OUTPUTS / f"exp_{eid}_sequential.csv").set_index("date")[["z", "obf_boundary"]])

with tab_calc:
    st.subheader("Power / sample size")
    p0 = st.number_input("Baseline conversion", 0.001, 0.99, 0.30, 0.01)
    mde = st.number_input("Relative MDE", 0.005, 1.0, 0.03, 0.005)
    st.metric("Users per arm", f"{experiments.sample_size_proportion(p0, mde):,}")
    st.subheader("Quick two-proportion test")
    a1, a2 = st.columns(2)
    xc = a1.number_input("Control conversions", 0, value=3000)
    nc = a1.number_input("Control users", 1, value=10000)
    xt = a2.number_input("Treatment conversions", 0, value=3150)
    nt = a2.number_input("Treatment users", 1, value=10000)
    r = S.two_proportion_ztest(xc, nc, xt, nt)
    st.write(f"Lift **{r['rel_lift']:+.2%}** (95% CI {r['rel_lift_ci'][0]:+.2%} … {r['rel_lift_ci'][1]:+.2%}), "
             f"p = {r['p_value']:.4f}; SRM p = {experiments.srm_test(nc, nt)['p_value']:.4f}")
