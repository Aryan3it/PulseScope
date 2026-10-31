"""Descriptive analytics: runs the SQL library and summarises distributions with inferential checks."""
from __future__ import annotations

import json

import numpy as np

from . import config as C
from . import sqlrun
from . import stats as S

SQL_REPORTS = ["funnel", "retention_cohorts", "engagement", "rfm", "category_mix"]


def run() -> dict:
    for name in SQL_REPORTS:
        sqlrun.run_file(name).to_csv(C.OUTPUTS / f"sql_{name}.csv", index=False)

    orders = sqlrun.query("SELECT revenue, platform, city_tier, discount_pct FROM orders")
    sessions = sqlrun.query("SELECT platform, city_tier, purchased, duration_min, views FROM sessions "
                            "USING SAMPLE 300000 ROWS (reservoir, 7)")
    uf = sqlrun.query("SELECT orders, revenue, sessions FROM user_features")

    out = {
        "describe": [S.describe(orders.revenue, "order_revenue"),
                     S.describe(uf.revenue, "revenue_per_user_60d"),
                     S.describe(uf.sessions, "sessions_per_user_60d"),
                     S.describe(sessions.duration_min, "session_minutes")],
        "aov_ci": dict(zip(["mean", "lo", "hi"], S.mean_ci(orders.revenue))),
        "conversion_wilson_ci": dict(zip(["p", "lo", "hi"], S.wilson_ci(int(sessions.purchased.sum()),
                                                                         len(sessions)))),
        "platform_vs_purchase": S.chi2_independence(sessions, "platform", "purchased"),
        "tier_vs_purchase": S.chi2_independence(sessions, "city_tier", "purchased"),
        "aov_app_vs_web": S.mann_whitney(orders.loc[orders.platform == "web", "revenue"],
                                         orders.loc[orders.platform != "web", "revenue"]),
        "discount_vs_revenue_spearman": float(orders[["discount_pct", "revenue"]].corr("spearman").iloc[0, 1]),
        "pareto_top20_revenue_share": float(np.sort(uf.revenue)[::-1][: len(uf) // 5].sum() / uf.revenue.sum()),
    }
    (C.OUTPUTS / "descriptive.json").write_text(json.dumps(out, indent=2, default=float))
    return out
