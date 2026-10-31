"""Behavioural user segmentation: feature engineering, model selection, persona naming,
validation against planted ground truth, and a Spark MLlib implementation for scale."""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

from . import config as C
from . import sqlrun

FEATURES = ["log_sessions", "log_active_days", "recency_days", "log_views_per_session", "cart_rate",
            "checkout_conv", "log_orders", "log_aov", "discount_share", "staples_view_share", "is_app"]

# Archetype scoring on standardized centroids -> human-readable segment names.
ARCHETYPES = {
    "High-Value Loyalists": lambda z: z.log_aov + z.log_orders + z.log_sessions - z.discount_share,
    "Deal Seekers": lambda z: 2 * z.discount_share + z.log_views_per_session + z.staples_view_share,
    "Mobile Impulse Buyers": lambda z: z.is_app - z.log_views_per_session + z.cart_rate,
    "Window Shoppers": lambda z: z.log_views_per_session - z.cart_rate - z.log_orders,
    "Checkout Abandoners": lambda z: -z.checkout_conv - z.log_orders,
    "Dormant / At-Risk": lambda z: -z.log_sessions - z.log_active_days + z.recency_days,
}


def build_features(uf: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame({"user_id": uf.user_id})
    f["log_sessions"] = np.log1p(uf.sessions)
    f["log_active_days"] = np.log1p(uf.active_days)
    f["recency_days"] = uf.recency_days
    f["log_views_per_session"] = np.log1p(uf.views_per_session)
    f["cart_rate"] = uf.cart_rate
    f["checkout_conv"] = uf.checkout_conv.fillna(uf.checkout_conv.median())
    f["log_orders"] = np.log1p(uf.orders)
    f["log_aov"] = np.log1p(uf.aov.fillna(0))
    f["discount_share"] = uf.discount_share.fillna(0)
    f["staples_view_share"] = uf.staples_view_share.fillna(uf.staples_view_share.median())
    f["is_app"] = uf.platform.isin(["android", "ios"]).astype(float)
    return f


def select_k(X: np.ndarray, ks=range(2, 10), sample: int = 15_000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    Xs = X[rng.choice(len(X), min(sample, len(X)), replace=False)]
    rows = []
    for k in ks:
        km = KMeans(k, n_init=5, random_state=seed).fit(Xs)
        gm = GaussianMixture(k, covariance_type="full", random_state=seed).fit(Xs)
        rows.append({"k": k, "inertia": km.inertia_, "silhouette": silhouette_score(Xs, km.labels_,
                                                                                    sample_size=5000,
                                                                                    random_state=seed),
                     "gmm_bic": gm.bic(Xs)})
    df = pd.DataFrame(rows)
    # Elbow on inertia via max distance to the chord between first and last k.
    x, y = df.k.to_numpy(float), df.inertia.to_numpy(float)
    xn, yn = (x - x[0]) / (x[-1] - x[0]), (y - y[-1]) / (y[0] - y[-1])
    df["elbow_distance"] = np.abs(xn + yn - 1) / np.sqrt(2)
    return df


def name_clusters(centroids_z: pd.DataFrame) -> dict[int, str]:
    names = list(ARCHETYPES)
    score = np.array([[ARCHETYPES[n](centroids_z.loc[c]) for n in names] for c in centroids_z.index])
    rows, cols = linear_sum_assignment(-score)
    mapping = {centroids_z.index[r]: names[c] for r, c in zip(rows, cols)}
    for c in centroids_z.index:
        mapping.setdefault(c, f"Segment {c}")
    return mapping


def stability(X: np.ndarray, k: int, n_runs: int = 5, sample: int = 20_000, seed: int = 0) -> float:
    """Mean pairwise ARI of K-Means fits on bootstrap resamples, evaluated on a common holdout."""
    rng = np.random.default_rng(seed)
    hold = X[rng.choice(len(X), min(sample, len(X)), replace=False)]
    labels = []
    for i in range(n_runs):
        boot = X[rng.integers(0, len(X), len(X))]
        labels.append(KMeans(k, n_init=3, random_state=i).fit(boot).predict(hold))
    return float(np.mean([adjusted_rand_score(labels[i], labels[j])
                          for i in range(n_runs) for j in range(i + 1, n_runs)]))


def spark_kmeans(k: int, seed: int = 0) -> tuple[pd.DataFrame, float]:
    """Same pipeline in Spark MLlib on the full table (assembler -> scaler -> k-means)."""
    from pyspark.ml import Pipeline
    from pyspark.ml.clustering import KMeans as SKMeans
    from pyspark.ml.feature import StandardScaler as SScaler
    from pyspark.ml.feature import VectorAssembler

    from .spark import get_spark

    spark = get_spark("pulsescope-segmentation", memory="4g")
    t0 = time.time()
    uf = spark.table(f"{C.HIVE_DB}.user_features").selectExpr(
        "user_id",
        "log1p(sessions) AS log_sessions", "log1p(active_days) AS log_active_days", "CAST(recency_days AS DOUBLE) AS recency_days",
        "log1p(views_per_session) AS log_views_per_session", "cart_rate",
        "checkout_conv", "log1p(orders) AS log_orders", "log1p(COALESCE(aov, 0)) AS log_aov",
        "COALESCE(discount_share, 0) AS discount_share", "staples_view_share",
        "CAST(platform IN ('android', 'ios') AS DOUBLE) AS is_app")
    med = uf.approxQuantile(["checkout_conv", "staples_view_share"], [0.5], 0.001)
    uf = uf.fillna({"checkout_conv": med[0][0], "staples_view_share": med[1][0]})
    pipe = Pipeline(stages=[VectorAssembler(inputCols=FEATURES, outputCol="raw"),
                            SScaler(inputCol="raw", outputCol="features", withMean=True, withStd=True),
                            SKMeans(k=k, seed=seed, featuresCol="features", predictionCol="spark_cluster",
                                    maxIter=50)])
    pred = pipe.fit(uf).transform(uf).select("user_id", "spark_cluster").toPandas()
    secs = time.time() - t0
    spark.stop()
    return pred, secs


def run(use_spark: bool = True) -> pd.DataFrame:
    uf = sqlrun.query("SELECT * FROM user_features")
    feats = build_features(uf)
    scaler = StandardScaler().fit(feats[FEATURES])
    X = scaler.transform(feats[FEATURES])

    sel = select_k(X)
    sel.to_csv(C.OUTPUTS / "segmentation_model_selection.csv", index=False)
    # Choose k at the inertia elbow, then confirm with GMM BIC flattening (reported, not tuned to truth).
    k = int(sel.loc[sel.elbow_distance.idxmax(), "k"])

    km = KMeans(k, n_init=20, random_state=0).fit(X)
    feats["cluster"] = km.labels_
    cz = pd.DataFrame(km.cluster_centers_, columns=FEATURES)
    names = name_clusters(cz)
    feats["segment"] = feats.cluster.map(names)

    prof = uf.assign(segment=feats.segment.values).groupby("segment").agg(
        users=("user_id", "size"), sessions=("sessions", "mean"), active_days=("active_days", "mean"),
        recency_days=("recency_days", "mean"), views_per_session=("views_per_session", "mean"),
        cart_rate=("cart_rate", "mean"), orders=("orders", "mean"), revenue=("revenue", "mean"),
        aov=("aov", "mean"), discount_share=("discount_share", "mean"),
        app_share=("platform", lambda s: s.isin(["android", "ios"]).mean()), churn_rate=("churned", "mean"))
    prof["user_share"] = prof.users / prof.users.sum()
    prof["revenue_share"] = prof.revenue * prof.users / (prof.revenue * prof.users).sum()
    prof = prof.sort_values("revenue_share", ascending=False)
    prof.to_csv(C.OUTPUTS / "segment_profiles.csv")

    # Interpretable surrogate: a depth-3 tree that explains segment membership in plain rules.
    tree = DecisionTreeClassifier(max_depth=4, min_samples_leaf=200, random_state=0).fit(feats[FEATURES],
                                                                                         feats.segment)
    rules = export_text(tree, feature_names=FEATURES, decimals=2)

    # Validation against the simulator's hidden personas.
    truth = pd.read_parquet(C.RAW / "users_truth.parquet")[["user_id", "persona"]]
    v = feats.merge(truth, on="user_id")
    rfm = sqlrun.run_file("rfm")[["user_id", "rfm_segment"]]
    v = v.merge(rfm, on="user_id", how="left").fillna({"rfm_segment": "non_buyer"})
    metrics = {
        "k": k, "n_users": len(feats),
        "silhouette": float(sel.loc[sel.k == k, "silhouette"].iloc[0]),
        "ari_vs_truth": adjusted_rand_score(v.persona, v.cluster),
        "nmi_vs_truth": normalized_mutual_info_score(v.persona, v.cluster),
        "ari_rfm_baseline": adjusted_rand_score(v.persona, v.rfm_segment),
        "nmi_rfm_baseline": normalized_mutual_info_score(v.persona, v.rfm_segment),
        "bootstrap_stability_ari": stability(X, k),
        "surrogate_tree_accuracy": float(tree.score(feats[FEATURES], feats.segment)),
        "cluster_names": {int(c): n for c, n in names.items()},
    }
    ct = pd.crosstab(v.segment, v.persona, normalize="index").round(3)
    ct.to_csv(C.OUTPUTS / "segment_vs_persona.csv")

    if use_spark:
        sp, secs = spark_kmeans(k)
        m = feats[["user_id", "cluster"]].merge(sp, on="user_id")
        metrics["spark_mllib_seconds"] = round(secs, 1)
        metrics["spark_vs_sklearn_ari"] = adjusted_rand_score(m.cluster, m.spark_cluster)

    (C.OUTPUTS / "segmentation.json").write_text(json.dumps(metrics, indent=2, default=float))
    (C.OUTPUTS / "segment_rules.txt").write_text(rules)
    feats[["user_id", "cluster", "segment"]].to_parquet(C.OUTPUTS / "segments.parquet", index=False)
    print(json.dumps(metrics, indent=2, default=float))
    return feats[["user_id", "cluster", "segment"]]


if __name__ == "__main__":
    run()
