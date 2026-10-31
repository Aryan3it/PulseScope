import numpy as np
import pandas as pd

from pulsescope import rca


def _slices(broken: str | None, cur_days):
    rows = []
    rng = np.random.default_rng(0)
    for day in range(28):
        for s, (sess, conv) in {"android": (4000, 0.5), "ios": (3000, 0.55), "web": (3000, 0.45)}.items():
            rate = conv * (0.2 if (s == broken and day in cur_days) else 1.0)
            checkouts = rng.binomial(sess, 0.3)
            rows.append({"day": day, "slice": s, "sessions": sess, "checkouts": checkouts,
                         "orders": rng.binomial(checkouts, rate)})
    return pd.DataFrame(rows)


def test_drilldown_attributes_drop_to_broken_slice():
    cur = [21, 22, 23]
    df = _slices("android", cur)
    dd = rca.drilldown(df, cur, [14, 15, 16, 7, 8, 9], "orders", "checkouts")
    top = dd.iloc[0]
    assert top.slice == "android"
    assert top.explanatory_power > 0.9
    ranking = rca.adtributor({"platform": dd})
    assert ranking.iloc[0].slices[0] == "android"


def test_mix_rate_decomposition_sums_to_total_change():
    cur, base = [21, 22], [14, 15]
    df = _slices("ios", cur)
    m = rca.mix_rate_decomposition(df, cur, base, "orders", "checkouts")
    c = df[df.day.isin(cur)].sum(numeric_only=True)
    b = df[df.day.isin(base)].sum(numeric_only=True)
    total = c.orders / c.checkouts - b.orders / b.checkouts
    assert np.isclose(m.values.sum(), total)


def test_detect_anomalies_finds_planted_dip():
    n = 70
    days = np.arange(n)
    base = 1000 + 100 * (days % 7 >= 5)
    sessions = base.astype(float)
    sessions[40:43] *= 0.6
    k = pd.DataFrame({"event_date": pd.date_range("2026-01-01", periods=n), "day": days, "sessions": sessions,
                      "dau": sessions * 0.8, "session_conversion": 0.1, "checkout_to_purchase": 0.7,
                      "revenue": sessions * 100, "aov": 1000.0})
    k[["session_conversion", "checkout_to_purchase", "aov"]] += np.random.default_rng(0).normal(0, 1e-3, (n, 3))
    a = rca.detect_anomalies(k)
    inc = rca.group_incidents(a)
    assert any(i["start_day"] <= 40 and i["end_day"] >= 42 for i in inc)


def test_version_rollover_is_not_blamed_for_volume_change():
    rows = []
    for day in range(28):
        ver = "3.9.0" if day >= 20 else "3.8.2"
        rows += [{"day": day, "slice": ver, "sessions": 3000}, {"day": day, "slice": "5.1.3", "sessions": 4000},
                 {"day": day, "slice": "email", "sessions": 2500 if day == 22 else 0}]
    df = pd.DataFrame(rows)
    df = df[df.sessions > 0]
    dd = rca.drilldown(df, [22], [15, 8, 1], "sessions", None)
    assert dd.iloc[0].slice == "email"
    assert "3.8.2→3.9.0" in set(dd.slice)
    assert abs(dd.set_index("slice").loc["3.8.2→3.9.0", "explanatory_power"]) < 0.01
