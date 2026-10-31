"""Clickstream simulator for a fictional e-commerce app ("ShopKart").

Writes a raw landing zone that looks like what a tracking SDK would dump:
    data/raw/events/dt=YYYY-MM-DD/part-0000.json.gz   (NDJSON, flat schema, with dirty rows)
    data/raw/users.csv                                 (CRM export)
and, separately, the *ground truth* that the analysis is later scored against:
    data/raw/ground_truth.json, data/raw/users_truth.parquet

Planted truth:
  * 5 latent behavioural personas (segmentation should recover them)
  * day 30: email campaign traffic spike (benign anomaly)
  * day 45: iOS 3.9.0 release with no effect (red herring)
  * days 60-66: Android 5.2.0 breaks checkout -> purchase for early-rollout users (the incident)
  * days 70-90: experiment A (one-click checkout), heterogeneous lift by persona
  * days 75-90: experiment B with a logging bug that drops iOS treatment exposures (SRM)
"""
from __future__ import annotations

import argparse
import json
import shutil

import numpy as np
import pandas as pd

from . import config as C

PERSONAS = np.array(["whale", "bargain_hunter", "impulse_mobile", "window_shopper", "dormant"])
PERSONA_P = [0.06, 0.20, 0.22, 0.30, 0.22]

#                      whale  bargain impulse window dormant
SESS_RATE = np.array([0.45, 0.30, 0.32, 0.18, 0.04])
VIEWS_MU = np.array([5.0, 9.0, 3.0, 7.0, 2.5])
P_CART = np.array([0.45, 0.32, 0.38, 0.10, 0.12])
P_CHECKOUT = np.array([0.78, 0.55, 0.70, 0.35, 0.45])
P_PURCHASE = np.array([0.88, 0.72, 0.70, 0.55, 0.62])
AOV_MEDIAN = np.array([4200.0, 850.0, 1100.0, 1600.0, 950.0])
AOV_SIGMA = np.array([0.50, 0.40, 0.45, 0.50, 0.50])
DISCOUNT_P = np.array([0.15, 0.85, 0.35, 0.25, 0.30])
CHURN_HALF_LIFE = np.array([600.0, 240.0, 150.0, 120.0, 45.0])
EXP_A_LIFT = np.array([1.02, 1.04, 1.15, 1.06, 1.00])   # multiplier on P(purchase | checkout)
EXP_B_LIFT = 1.05                                        # multiplier on P(cart | session)

PLATFORMS = np.array(["android", "ios", "web"])
PLATFORM_P = np.array([
    [0.25, 0.45, 0.30],
    [0.55, 0.15, 0.30],
    [0.70, 0.28, 0.02],
    [0.35, 0.20, 0.45],
    [0.45, 0.20, 0.35],
])
CHANNELS = np.array(["organic", "paid_search", "social", "referral", "email"])
CHANNEL_P = np.array([
    [0.35, 0.20, 0.05, 0.20, 0.20],
    [0.25, 0.30, 0.25, 0.10, 0.10],
    [0.20, 0.15, 0.50, 0.10, 0.05],
    [0.40, 0.25, 0.20, 0.10, 0.05],
    [0.30, 0.30, 0.25, 0.10, 0.05],
])
REGIONS = np.array(["north", "south", "east", "west"])
TIERS = np.array(["metro", "tier2", "tier3"])
TIER_P = np.array([
    [0.75, 0.20, 0.05],
    [0.30, 0.40, 0.30],
    [0.45, 0.35, 0.20],
    [0.50, 0.30, 0.20],
    [0.35, 0.35, 0.30],
])
CATEGORIES = np.array(["electronics", "fashion", "home", "beauty", "grocery", "sports"])
CATEGORY_P = np.array([
    [0.35, 0.25, 0.20, 0.10, 0.02, 0.08],
    [0.10, 0.30, 0.15, 0.15, 0.25, 0.05],
    [0.20, 0.35, 0.05, 0.25, 0.05, 0.10],
    [0.30, 0.25, 0.20, 0.10, 0.05, 0.10],
    [0.15, 0.20, 0.20, 0.15, 0.20, 0.10],
])
HOUR_P = np.array([1, 1, 1, 1, 1, 1, 2, 3, 4, 4, 4, 4, 5, 5, 4, 4, 4, 5, 6, 8, 9, 9, 7, 4], dtype=float)
HOUR_P /= HOUR_P.sum()


def _choice_rows(rng, probs_by_row: np.ndarray, row_idx: np.ndarray) -> np.ndarray:
    """Vectorised categorical draw where each element uses probs_by_row[row_idx[i]]."""
    cdf = probs_by_row.cumsum(axis=1)[row_idx]
    u = rng.random(len(row_idx))[:, None]
    return (u > cdf).sum(axis=1)


def make_users(n_users: int, rng) -> pd.DataFrame:
    persona = rng.choice(len(PERSONAS), size=n_users, p=PERSONA_P)
    # Signups ramp up over time (growing product), some predate the window.
    signup_day = np.floor(-120 + 205 * rng.beta(1.3, 1.0, n_users)).astype(int)
    churn_day = signup_day + rng.exponential(CHURN_HALF_LIFE[persona] / np.log(2))
    platform = _choice_rows(rng, PLATFORM_P, persona)
    return pd.DataFrame({
        "user_id": np.arange(1, n_users + 1),
        "persona_idx": persona,
        "signup_day": signup_day,
        "churn_day": churn_day,
        "engagement": rng.gamma(2.0, 0.5, n_users),
        "platform_idx": platform,
        "android_early": (platform == 0) & (rng.random(n_users) < 0.40),
        "channel_idx": _choice_rows(rng, CHANNEL_P, persona),
        "region_idx": rng.integers(0, len(REGIONS), n_users),
        "tier_idx": _choice_rows(rng, TIER_P, persona),
        "exp_a_treat": rng.random(n_users) < 0.5,
        "exp_b_treat": rng.random(n_users) < 0.5,
    })


def app_version(platform_idx: np.ndarray, android_early: np.ndarray, day: int) -> np.ndarray:
    v = np.empty(len(platform_idx), dtype=object)
    android, ios = platform_idx == 0, platform_idx == 1
    v[platform_idx == 2] = "web"
    v[ios] = "3.9.0" if day >= C.IOS_RELEASE_DAY else "3.8.2"
    v[android] = "5.1.3"
    if C.INCIDENT_START <= day < C.INCIDENT_END:
        v[android & android_early] = "5.2.0"
    elif day >= C.INCIDENT_END:
        v[android & android_early] = "5.2.1"
    return v


def simulate_day(day: int, users: pd.DataFrame, rng) -> pd.DataFrame:
    u = users
    active = (u.signup_day.values <= day) & (u.churn_day.values > day)
    weekday = C.day_to_date(day).weekday()
    seasonal = (1.20 if weekday >= 5 else 1.0) * (1 + 0.002 * day)
    lam = SESS_RATE[u.persona_idx.values] * u.engagement.values * seasonal * active
    n_sess = np.minimum(rng.poisson(lam), 4)

    campaign = np.zeros(len(u), dtype=int)
    if day == C.CAMPAIGN_DAY:
        reached = active & (rng.random(len(u)) < 0.35)
        campaign = rng.binomial(1, 0.6, len(u)) * reached
    total = n_sess + campaign
    sidx = np.repeat(np.arange(len(u)), total)
    is_campaign = np.concatenate([np.r_[np.zeros(a, bool), np.ones(b, bool)]
                                  for a, b in zip(n_sess[total > 0], campaign[total > 0])]) if campaign.any() \
        else np.zeros(len(sidx), bool)
    S = len(sidx)
    if S == 0:
        return pd.DataFrame()

    p = u.persona_idx.values[sidx]
    plat = u.platform_idx.values[sidx]
    version = app_version(plat, u.android_early.values[sidx], day)

    in_a = (C.EXP_A["start"] <= day < C.EXP_A["end"]) & (u.signup_day.values[sidx] <= C.EXP_A["start"])
    treat_a = in_a & u.exp_a_treat.values[sidx]
    in_b = (C.EXP_B["start"] <= day < C.EXP_B["end"]) & (u.signup_day.values[sidx] <= C.EXP_B["start"])
    treat_b = in_b & u.exp_b_treat.values[sidx]

    p_cart = P_CART[p] * np.where(treat_b, EXP_B_LIFT, 1.0) * np.where(is_campaign, 1.3, 1.0)
    p_purchase = P_PURCHASE[p] * np.where(treat_a, EXP_A_LIFT[p], 1.0)
    p_purchase = np.where(version == "5.2.0", p_purchase * 0.20, p_purchase)

    views = 1 + rng.poisson(VIEWS_MU[p] - 1)
    cart = rng.random(S) < p_cart
    checkout = cart & (rng.random(S) < P_CHECKOUT[p] * np.where(treat_a, 1.03, 1.0))
    purchase = checkout & (rng.random(S) < np.minimum(p_purchase, 0.99))
    category = _choice_rows(rng, CATEGORY_P, p)
    price = np.round(rng.lognormal(np.log(AOV_MEDIAN[p]), AOV_SIGMA[p]), 2)
    discount = np.where(rng.random(S) < DISCOUNT_P[p], rng.choice([0.1, 0.2, 0.3], S), 0.0)
    revenue = np.round(price * (1 - discount), 2)

    hour = rng.choice(24, S, p=HOUR_P)
    sess_ts = (pd.Timestamp(C.day_to_date(day)) + pd.to_timedelta(hour * 3600 + rng.integers(0, 3600, S), unit="s"))
    gap = rng.integers(8, 60, S)
    source = np.where(is_campaign, "email", np.array(["direct", "organic_search", "paid", "push"])[
        rng.choice(4, S, p=[0.40, 0.30, 0.20, 0.10])])
    session_id = np.char.add(f"s{day:03d}-", np.arange(S).astype(str))

    # Experiment B logging bug: half of iOS treatment exposures never reach the pipeline.
    logged_b = in_b & ~(treat_b & (plat == 1) & (rng.random(S) < 0.5))

    frames = []

    def emit(mask, name, seq, **extra):
        idx = np.flatnonzero(mask)
        frames.append(pd.DataFrame({"_s": idx, "_seq": seq[idx] if isinstance(seq, np.ndarray) else seq,
                                    "event_name": name, **{k: v[idx] for k, v in extra.items()}}))

    emit(np.ones(S, bool), "session_start", 0)
    emit(in_a, "experiment_exposure", 1, experiment_id=np.full(S, C.EXP_A["id"], object),
         variant=np.where(treat_a, "treatment", "control").astype(object))
    emit(logged_b, "experiment_exposure", 1, experiment_id=np.full(S, C.EXP_B["id"], object),
         variant=np.where(treat_b, "treatment", "control").astype(object))
    vs = np.repeat(np.arange(S), views)
    within = np.arange(len(vs)) - np.repeat(np.cumsum(views) - views, views)
    vcat = np.where(rng.random(len(vs)) < 0.6, category[vs], rng.integers(0, len(CATEGORIES), len(vs)))
    frames.append(pd.DataFrame({"_s": vs, "_seq": 2 + within, "event_name": "view_item",
                                "category": CATEGORIES[vcat],
                                "price": np.round(rng.lognormal(np.log(AOV_MEDIAN[p[vs]]), 0.7), 2)}))
    emit(cart, "add_to_cart", 2 + views, category=CATEGORIES[category], price=price)
    emit(checkout, "begin_checkout", 3 + views, price=price)
    order_id = np.char.add(f"o{day:03d}-", np.arange(S).astype(str))
    emit(purchase, "purchase", 4 + views, category=CATEGORIES[category], price=price,
         order_id=order_id, revenue=revenue, discount_pct=discount)

    ev = pd.concat(frames, ignore_index=True).sort_values(["_s", "_seq"], kind="stable")
    s = ev["_s"].to_numpy()
    uid = u.user_id.values[sidx][s]
    ev.insert(0, "event_ts", (sess_ts[s] + pd.to_timedelta(ev["_seq"].to_numpy() * gap[s], unit="s"))
              .strftime("%Y-%m-%dT%H:%M:%S"))
    ev.insert(1, "user_id", uid)
    ev.insert(2, "session_id", session_id[s])
    ev["platform"] = PLATFORMS[plat[s]]
    ev["app_version"] = version[s]
    ev["geo_region"] = REGIONS[u.region_idx.values[sidx][s]]
    ev["city_tier"] = TIERS[u.tier_idx.values[sidx][s]]
    ev["traffic_source"] = source[s]
    ev = ev.drop(columns=["_s", "_seq"]).reset_index(drop=True)
    ev.insert(0, "event_id", np.char.add(f"e{day:03d}-", np.arange(len(ev)).astype(str)))

    # Dirty data: SDK retries (duplicates), missing user ids, internal QA traffic.
    n = len(ev)
    dup = ev.iloc[rng.choice(n, int(n * 0.004), replace=False)]
    ev = pd.concat([ev, dup], ignore_index=True)
    ev["user_id"] = ev["user_id"].astype("Int64")
    ev.loc[rng.random(len(ev)) < 0.0005, "user_id"] = pd.NA
    qa = ev.sample(n=min(200, len(ev)), random_state=day).copy()
    qa["user_id"] = -rng.integers(1, 20, len(qa))
    return pd.concat([ev, qa], ignore_index=True)


def main(n_users: int = 100_000, seed: int = 7) -> dict:
    rng = np.random.default_rng(seed)
    users = make_users(n_users, rng)
    if C.RAW.exists():
        shutil.rmtree(C.RAW)
    C.RAW_EVENTS.mkdir(parents=True)

    total_events = 0
    for day in range(C.N_DAYS):
        ev = simulate_day(day, users, rng)
        out = C.RAW_EVENTS / f"dt={C.date_str(day)}"
        out.mkdir()
        ev.to_json(out / "part-0000.json.gz", orient="records", lines=True, compression="gzip")
        total_events += len(ev)
        if day % 10 == 0:
            print(f"  day {day:3d}: {len(ev):,} events")

    crm = pd.DataFrame({
        "user_id": users.user_id,
        "signup_date": [C.date_str(d) for d in users.signup_day],
        "primary_platform": PLATFORMS[users.platform_idx],
        "acquisition_channel": CHANNELS[users.channel_idx],
        "region": REGIONS[users.region_idx],
        "city_tier": TIERS[users.tier_idx],
    })
    crm.to_csv(C.RAW / "users.csv", index=False)

    truth_users = pd.DataFrame({
        "user_id": users.user_id,
        "persona": PERSONAS[users.persona_idx],
        "churn_day": users.churn_day.round(1),
        "engagement": users.engagement.round(3),
        "exp_a_variant": np.where(users.exp_a_treat, "treatment", "control"),
        "exp_b_variant": np.where(users.exp_b_treat, "treatment", "control"),
    })
    truth_users.to_parquet(C.RAW / "users_truth.parquet", index=False)
    truth = {
        "n_users": n_users, "seed": seed, "raw_events": total_events,
        "campaign": {"date": C.date_str(C.CAMPAIGN_DAY), "dimension": "traffic_source", "value": "email"},
        "red_herring": {"date": C.date_str(C.IOS_RELEASE_DAY), "platform": "ios", "app_version": "3.9.0"},
        "incident": {"start": C.date_str(C.INCIDENT_START), "end": C.date_str(C.INCIDENT_END - 1),
                     "platform": "android", "app_version": "5.2.0", "funnel_step": "checkout_to_purchase",
                     "effect": "P(purchase|checkout) x0.20"},
        "experiment_a": {**C.EXP_A, "metric_effect": "P(purchase|checkout) multiplier by persona",
                         "lift_by_persona": dict(zip(PERSONAS.tolist(), EXP_A_LIFT.tolist()))},
        "experiment_b": {**C.EXP_B, "bug": "50% of iOS treatment exposures dropped -> SRM",
                         "cart_lift": EXP_B_LIFT},
        "personas": dict(zip(PERSONAS.tolist(), PERSONA_P)),
    }
    (C.RAW / "ground_truth.json").write_text(json.dumps(truth, indent=2))
    print(f"wrote {total_events:,} raw events for {n_users:,} users -> {C.RAW_EVENTS}")
    return truth


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    main(a.users, a.seed)
