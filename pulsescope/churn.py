"""Churn-propensity modelling with H2O AutoML, benchmarked against scikit-learn baselines."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

from . import config as C
from . import sqlrun

NUMERIC = ["sessions", "active_days", "recency_days", "active_span_days", "views_per_session",
           "avg_session_min", "carts", "orders", "revenue", "aov", "cart_rate", "checkout_conv",
           "discount_share", "weekend_share", "n_categories", "staples_view_share", "tenure_days"]
CATEGORICAL = ["platform", "acquisition_channel", "region", "city_tier", "segment"]
TARGET = "churned"


def load(segments: pd.DataFrame | None) -> pd.DataFrame:
    df = sqlrun.query("SELECT * FROM user_features")
    if segments is not None:
        df = df.merge(segments[["user_id", "segment"]], on="user_id", how="left")
    else:
        df["segment"] = "all"
    return df


def lift_at(y, p, frac: float = 0.1) -> float:
    k = int(len(y) * frac)
    top = np.argsort(-p)[:k]
    return float(np.mean(np.asarray(y)[top]) / np.mean(y))


def sklearn_baselines(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    pre = ColumnTransformer([("num", make_pipeline(StandardScaler()), NUMERIC),
                             ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)])
    Xtr, Xte = train[NUMERIC + CATEGORICAL].copy(), test[NUMERIC + CATEGORICAL].copy()
    med = Xtr[NUMERIC].median()
    Xtr[NUMERIC], Xte[NUMERIC] = Xtr[NUMERIC].fillna(med), Xte[NUMERIC].fillna(med)
    out = {}
    for name, model in {"logistic_regression": LogisticRegression(max_iter=2000, C=1.0),
                        "hist_gradient_boosting": HistGradientBoostingClassifier(random_state=0)}.items():
        pipe = make_pipeline(pre, model).fit(Xtr, train[TARGET])
        p = pipe.predict_proba(Xte)[:, 1]
        out[name] = {"auc": roc_auc_score(test[TARGET], p), "aucpr": average_precision_score(test[TARGET], p),
                     "brier": brier_score_loss(test[TARGET], p), "lift_top_decile": lift_at(test[TARGET], p)}
    return out


def h2o_automl(train: pd.DataFrame, test: pd.DataFrame, max_runtime: int = 180) -> tuple[dict, np.ndarray]:
    import h2o
    from h2o.automl import H2OAutoML

    h2o.init(max_mem_size="4G", nthreads=-1, verbose=False)
    h2o.no_progress()
    cols = NUMERIC + CATEGORICAL + [TARGET]
    htr, hte = h2o.H2OFrame(train[cols]), h2o.H2OFrame(test[cols])
    for f in (htr, hte):
        f[TARGET] = f[TARGET].asfactor()
        for c in CATEGORICAL:
            f[c] = f[c].asfactor()
    aml = H2OAutoML(max_runtime_secs=max_runtime, max_models=15, seed=1, sort_metric="AUC",
                    exclude_algos=["DeepLearning"], nfolds=5)
    aml.train(x=NUMERIC + CATEGORICAL, y=TARGET, training_frame=htr)
    lb = aml.leaderboard.as_data_frame()
    lb.to_csv(C.OUTPUTS / "churn_h2o_leaderboard.csv", index=False)
    p = aml.leader.predict(hte)["p1"].as_data_frame()["p1"].to_numpy()

    # Variable importance from the best non-ensemble model (ensembles do not expose it).
    vi = None
    for mid in lb.model_id:
        if not mid.startswith("StackedEnsemble"):
            vi = h2o.get_model(mid).varimp(use_pandas=True)
            vi_model = mid
            break
    if vi is not None:
        vi.to_csv(C.OUTPUTS / "churn_feature_importance.csv", index=False)
    res = {"leader": aml.leader.model_id, "n_models": len(lb),
           "auc": roc_auc_score(test[TARGET], p), "aucpr": average_precision_score(test[TARGET], p),
           "brier": brier_score_loss(test[TARGET], p), "lift_top_decile": lift_at(test[TARGET], p),
           "importance_model": vi_model if vi is not None else None,
           "top_features": vi.variable.head(8).tolist() if vi is not None else []}
    h2o.cluster().shutdown()
    return res, p


def run(segments: pd.DataFrame | None = None, max_runtime: int = 180) -> dict:
    df = load(segments)
    train, test = train_test_split(df, test_size=0.2, stratify=df[TARGET], random_state=0)
    out = {"n_train": len(train), "n_test": len(test), "churn_rate": float(df[TARGET].mean()),
           "baselines": sklearn_baselines(train, test)}
    try:
        out["h2o_automl"], p = h2o_automl(train, test, max_runtime)
    except Exception as e:  # H2O needs a JVM; keep the pipeline usable without it.
        out["h2o_automl"] = {"error": repr(e)}
        p = None
    if p is not None:
        scored = test[["user_id", "segment", "revenue", TARGET]].assign(churn_prob=p)
        risk = scored.groupby("segment").agg(users=("user_id", "size"), predicted_churn=("churn_prob", "mean"),
                                             actual_churn=(TARGET, "mean"), avg_revenue=("revenue", "mean"))
        risk["revenue_at_risk_per_100_users"] = 100 * risk.predicted_churn * risk.avg_revenue
        risk.sort_values("revenue_at_risk_per_100_users", ascending=False).to_csv(
            C.OUTPUTS / "churn_risk_by_segment.csv")
        scored.to_parquet(C.OUTPUTS / "churn_scores_test.parquet", index=False)
    (C.OUTPUTS / "churn.json").write_text(json.dumps(out, indent=2, default=float))
    print(json.dumps(out, indent=2, default=float))
    return out
