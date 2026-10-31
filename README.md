# PulseScope — Product Analytics, Segmentation & Experimentation Platform

An end-to-end product data-science platform over the clickstream of a simulated e-commerce app
(**ShopKart**, ~100k users, ~12M raw events over 91 days). It ingests raw event logs with **Spark** into a
**Hive-metastore** Parquet lake, answers product questions in **SQL**, segments users with **ML**, predicts
churn with **H2O AutoML**, runs **A/B tests** with CUPED and sequential monitoring (independently re-checked in
**R**), **root-causes KPI drops**, serves segment profiles from **MongoDB**, and turns it all into an insight
report and a **Streamlit** dashboard.

The twist that makes it evaluable: the simulator **plants ground truth** (a production incident, a benign
campaign spike, a red-herring release, hidden personas, a true experiment effect, and an experiment with a logging
bug). Every method is scored on whether it recovers that truth, not just on whether it runs.

## Results (default run: 100k users, seed 7)

| Area | Result |
|---|---|
| **Spark ETL** | 10.78M raw events → 10.72M clean in ~76 s on a laptop (local[*]); removed 42.8k duplicate events, 5.5k null user ids, 18.2k internal-QA rows; 8 Hive tables (1.33M sessions, 194k orders) |
| **Root-cause analysis** | Flagged exactly 2 incidents, 0 false alarms. Incident window matched exactly (May 31 – Jun 6); localised to checkout→purchase (−15.8%, 98% of the conversion drop); root cause `app_version = 5.2.0` (≈100% explanatory power); mix-vs-rate shows a pure rate effect. Campaign spike attributed to `traffic_source = email` (86%). Red-herring iOS release not blamed |
| **Experiment A** (one-click checkout) | Conversion +3.54% (95% CI +1.41% … +5.72%, p = 0.001) → **SHIP**. Revenue/user: raw test p = 0.50, **CUPED cut variance 57%** (CI 34% narrower) → +₹131/user, p = 0.0006. O'Brien-Fleming boundary crossed on day 6 of 21. Largest segment effect in High-Value Loyalists (+5.4%, BH-adjusted p = 0.0001) |
| **Experiment B** (reco widget) | SRM detected (χ² = 27.8, p < 1e-6) and localised to iOS (45.5% treatment share; web/Android ≈ 50%) → **INVALID**, results not read |
| **Test calibration** | A/A false-positive rate 4.4% over 1,000 shuffles (p-values uniform, KS p = 0.93). Peeking at 10 looks: naive FPR 19.4% vs. O'Brien-Fleming 4.6% |
| **R cross-check** | 12 quantities re-computed in R; Python and R agree to < 1e-10 relative error (CUPED vs. ANCOVA 9e-5, as expected) |
| **Segmentation** | k = 5 chosen by inertia elbow (no labels). ARI vs. hidden personas **0.35 — 2× the RFM baseline (0.18)**; bootstrap stability ARI 0.79; depth-4 surrogate tree explains 91.7% of assignments. Spark MLlib K-Means on the full table in 4–12 s (ARI 0.76 vs. scikit-learn). Found a product-relevant *Checkout Abandoners* segment (7% of users, 0.01 orders/user despite cart activity) |
| **Churn (H2O AutoML)** | 12 models in 3 min; best GBM test AUC **0.913**, AUC-PR 0.814, 3.76× lift in top decile; ties tuned HistGBM (0.913), beats logistic regression (0.900). Top drivers: recency, segment, active days |
| **MongoDB** | 61,750 user profiles + 5 segment profiles upserted, indexed on (segment, churn_prob) |
| **Tests** | 19 unit tests (stats checked against scipy/statsmodels, OBF boundaries vs. published values, CUPED unbiasedness, RCA attribution, rollover handling, simulator) |

**Honest limitations.** The data is simulated, so effect sizes come from the design, not the market. The smallest
persona (whales, 6%) merges with frequent impulse buyers at k = 5, which is why ARI tops out at 0.35. Experiment A's
per-segment lifts differ but the formal heterogeneity test is not significant (Cochran's Q p = 0.63), so the report
does not claim heterogeneous effects. Anomaly detection needs a materiality floor: z-scores alone raised 7 false alarms
on sub-3% traffic moves.

## Architecture

```
simulate.py ──► data/raw/events/dt=*/part.json.gz   (NDJSON, duplicates, null ids, QA traffic)
                data/raw/users.csv                    (CRM export)
                data/raw/ground_truth.json            (hidden; used only for scoring)
      │
      ▼  etl.py  (PySpark + Hive metastore)
data/warehouse/pulsescope.db/   events · users · sessions · orders · daily_kpis ·
                                experiment_assignments · experiment_users · user_features
      │
      ├─► analytics.py + sql/*.sql (DuckDB)  funnels, cohort retention, DAU/WAU/MAU, RFM, category mix
      ├─► segmentation.py   sklearn K-Means/GMM + Spark MLlib K-Means, persona naming, surrogate tree
      ├─► churn.py          H2O AutoML vs. logistic regression / HistGBM baselines
      ├─► experiments.py    power, SRM, z/Welch tests, CUPED, bootstrap, HTE + BH-FDR, O'Brien-Fleming, A/A
      │      └─► r/ab_validation.R   independent re-analysis (prop.test, t.test, ANCOVA, boot)
      ├─► rca.py            STL anomaly detection → funnel localisation → Adtributor drill-down → mix/rate
      ├─► nosql.py          user + segment profiles → MongoDB (indexed, aggregation pipeline)
      └─► report.py         outputs/report.html   ·   dashboard/app.py (Streamlit)
```

## Planted scenario

| When | What happens | What the analysis must do |
|---|---|---|
| all period | 5 latent personas (whale, bargain hunter, impulse mobile, window shopper, dormant) | recover them without labels |
| day 30 | email campaign → traffic spike | flag it, attribute to `traffic_source = email`, not call it an incident |
| day 45 | iOS 3.9.0 release, no effect | **not** blame it |
| days 60–66 | Android 5.2.0 (40% of Android users) breaks checkout→purchase (×0.2) | detect window, localise funnel step, find `app_version = 5.2.0` |
| days 70–90 | Experiment A: one-click checkout, lift varies by persona (0–15%) | detect lift, quantify HTE with FDR control, ship decision |
| days 75–90 | Experiment B: 50% of iOS treatment exposures dropped by a logging bug | catch the sample-ratio mismatch and localise it to iOS |

## Techniques by area

- **Big data / ETL** — PySpark schema-on-read, data-quality audit, dedup, partitioned Parquet, Hive tables, Spark
  SQL marts (sessionisation, experiment metrics with pre-period covariates, user features + churn label), Spark MLlib.
- **SQL** — window functions (`LAG`, `FIRST_VALUE`, `NTILE`, `RANK`), cohort retention, rolling WAU/MAU,
  RFM, funnel conversion, interim-look experiment queries (`sql/`).
- **Descriptive statistics** — distribution summaries (skew, kurtosis, Pareto share), Wilson and t intervals.
- **Inferential statistics** — two-proportion z, Welch t, χ² independence + Cramér's V, Mann-Whitney, bootstrap,
  Benjamini-Hochberg, Cochran's Q heterogeneity.
- **Experimentation** — sample size / MDE, SRM detection and localisation, CUPED, guardrails, HTE, O'Brien-Fleming
  boundaries calibrated by Monte Carlo, A/A calibration, peeking simulation, cross-language validation in R.
- **ML** — K-Means / GMM model selection (elbow, silhouette, BIC), Hungarian-matched persona naming, bootstrap
  stability, decision-tree surrogate rules, H2O AutoML (GBM, XGBoost, GLM, DRF, stacked ensembles).
- **Root-cause analysis** — STL + robust z anomaly detection, incident grouping, same-weekday baselines, log
  funnel decomposition, Adtributor explanatory power / surprise, two-level drill-down, mix-vs-rate decomposition.

## Running it

Requires Python 3.10+, Java 8/11/17 (Spark and H2O), and R ≥ 4 with `dplyr`, `jsonlite`, `boot`. Docker is optional (MongoDB).

```bash
make setup                 # venv + requirements
make mongo                 # optional: MongoDB in Docker
make all                   # simulate → etl → analytics → segment → churn → experiments → rca → R → nosql → report
make all USERS=20000       # smaller/faster run
make test                  # unit tests
make dashboard             # Streamlit UI
```

Each stage can run alone: `python -m pulsescope.pipeline rca report`.

## Repository layout

```
pulsescope/   simulate · spark · etl · sqlrun · analytics · stats · experiments · segmentation · churn · rca · nosql · report · pipeline
pulsescope/sql/   analytical SQL library
r/            ab_validation.R
dashboard/    Streamlit app
tests/        unit tests (stats vs scipy/statsmodels, OBF boundaries, CUPED, RCA attribution, simulator)
```
