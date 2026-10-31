"""Spark ETL: raw NDJSON landing zone -> validated Parquet lake registered in the Hive metastore.

Tables (database `pulsescope`):
  events                 cleaned, de-duplicated events, partitioned by event_date
  users                  CRM dimension
  sessions               one row per session with funnel flags and context
  orders                 one row per purchase
  daily_kpis             company-level daily KPIs
  experiment_assignments first exposure per (experiment, user)
  experiment_users       per-user experiment metrics + pre-period covariates (CUPED)
  user_features          behavioural features over the observation window + churn label
"""
from __future__ import annotations

import json
import time

from pyspark.sql import functions as F
from pyspark.sql import types as T

from . import config as C
from .spark import get_spark

RAW_SCHEMA = T.StructType([
    T.StructField("event_id", T.StringType()),
    T.StructField("event_ts", T.StringType()),
    T.StructField("user_id", T.LongType()),
    T.StructField("session_id", T.StringType()),
    T.StructField("event_name", T.StringType()),
    T.StructField("experiment_id", T.StringType()),
    T.StructField("variant", T.StringType()),
    T.StructField("category", T.StringType()),
    T.StructField("price", T.DoubleType()),
    T.StructField("order_id", T.StringType()),
    T.StructField("revenue", T.DoubleType()),
    T.StructField("discount_pct", T.DoubleType()),
    T.StructField("platform", T.StringType()),
    T.StructField("app_version", T.StringType()),
    T.StructField("geo_region", T.StringType()),
    T.StructField("city_tier", T.StringType()),
    T.StructField("traffic_source", T.StringType()),
])
VALID_EVENTS = ["session_start", "experiment_exposure", "view_item", "add_to_cart", "begin_checkout", "purchase"]


def save(df, name: str, partition_by: str | None = None):
    w = df.write.mode("overwrite").format("parquet")
    if partition_by:
        w = w.partitionBy(partition_by)
    w.saveAsTable(f"{C.HIVE_DB}.{name}")


def run():
    t0 = time.time()
    spark = get_spark("pulsescope-etl")
    spark.sql(f"CREATE DATABASE IF NOT EXISTS {C.HIVE_DB} LOCATION '{C.LAKE}'")
    spark.sql(f"USE {C.HIVE_DB}")

    # ---------- 1. ingest + data quality ----------
    raw = spark.read.schema(RAW_SCHEMA).json(str(C.RAW_EVENTS))
    raw = raw.withColumn("ts", F.to_timestamp("event_ts", "yyyy-MM-dd'T'HH:mm:ss")).cache()
    dq = raw.agg(
        F.count("*").alias("raw_rows"),
        F.sum(F.col("user_id").isNull().cast("int")).alias("null_user_id"),
        F.sum((F.col("user_id") < 0).cast("int")).alias("internal_qa_rows"),
        F.sum(F.col("ts").isNull().cast("int")).alias("bad_timestamp"),
        F.sum((~F.col("event_name").isin(VALID_EVENTS)).cast("int")).alias("unknown_event_name"),
    ).first().asDict()

    valid = raw.filter(F.col("user_id").isNotNull() & (F.col("user_id") > 0) & F.col("ts").isNotNull()
                       & F.col("event_name").isin(VALID_EVENTS))
    dq["duplicate_event_ids"] = valid.count() - valid.select("event_id").distinct().count()
    events = (valid.dropDuplicates(["event_id"])
              .withColumn("event_date", F.to_date("ts"))
              .withColumn("day", F.datediff("event_date", F.lit(C.START_DATE.isoformat())))
              .drop("event_ts").withColumnRenamed("ts", "event_ts"))
    save(events.repartition("event_date"), "events", "event_date")
    events = spark.table("events")
    dq["clean_rows"] = events.count()
    dq["dropped_rows"] = dq["raw_rows"] - dq["clean_rows"]
    raw.unpersist()

    users = (spark.read.option("header", True).option("inferSchema", True).csv(str(C.RAW / "users.csv"))
             .withColumn("signup_date", F.to_date("signup_date")))
    save(users, "users")

    # ---------- 2. marts (Spark SQL) ----------
    sessions = spark.sql("""
        SELECT session_id, user_id,
               MIN(event_ts)                                         AS session_start,
               MIN(event_date)                                       AS event_date,
               MIN(day)                                              AS day,
               FIRST(platform)  AS platform,  FIRST(app_version) AS app_version,
               FIRST(geo_region) AS geo_region, FIRST(city_tier) AS city_tier,
               FIRST(traffic_source) AS traffic_source,
               SUM(CAST(event_name = 'view_item' AS INT))            AS views,
               MAX(CAST(event_name = 'add_to_cart' AS INT))          AS carted,
               MAX(CAST(event_name = 'begin_checkout' AS INT))       AS checkout,
               MAX(CAST(event_name = 'purchase' AS INT))             AS purchased,
               COALESCE(SUM(CASE WHEN event_name = 'purchase' THEN revenue END), 0) AS revenue,
               MAX(CASE WHEN event_name = 'purchase' THEN discount_pct END)         AS discount_pct,
               (UNIX_TIMESTAMP(MAX(event_ts)) - UNIX_TIMESTAMP(MIN(event_ts))) / 60.0 AS duration_min
        FROM events GROUP BY session_id, user_id
    """)
    save(sessions, "sessions", "event_date")

    orders = spark.sql("""
        SELECT order_id, user_id, session_id, event_ts AS order_ts, event_date, day, category,
               price AS list_price, discount_pct, revenue, platform, app_version, geo_region, city_tier
        FROM events WHERE event_name = 'purchase'
    """)
    save(orders, "orders")

    save(spark.sql("""
        SELECT event_date, day, COUNT(DISTINCT user_id) AS dau, COUNT(*) AS sessions,
               SUM(carted) AS carts, SUM(checkout) AS checkouts, SUM(purchased) AS orders,
               ROUND(SUM(revenue), 2) AS revenue,
               SUM(purchased) / COUNT(*) AS session_conversion,
               SUM(purchased) / NULLIF(SUM(checkout), 0) AS checkout_to_purchase,
               SUM(revenue) / NULLIF(SUM(purchased), 0) AS aov
        FROM sessions GROUP BY event_date, day ORDER BY event_date
    """), "daily_kpis")

    save(spark.sql("""
        SELECT experiment_id, user_id,
               MIN_BY(variant, event_ts)          AS variant,
               COUNT(DISTINCT variant)            AS n_variants,
               MIN(event_ts)                      AS first_exposure_ts,
               MIN(day)                           AS first_exposure_day,
               FIRST(platform)                    AS platform
        FROM events WHERE event_name = 'experiment_exposure'
        GROUP BY experiment_id, user_id
    """), "experiment_assignments")

    pre_lo, pre_hi = C.CUPED_PRE_WINDOW
    save(spark.sql(f"""
        WITH post AS (
            SELECT a.experiment_id, a.user_id,
                   COUNT(s.session_id)                 AS sessions,
                   COALESCE(SUM(s.carted), 0)          AS carts,
                   COALESCE(SUM(s.checkout), 0)        AS checkouts,
                   COALESCE(SUM(s.purchased), 0)       AS orders,
                   COALESCE(SUM(s.revenue), 0)         AS revenue
            FROM experiment_assignments a
            LEFT JOIN sessions s ON s.user_id = a.user_id AND s.session_start >= a.first_exposure_ts
            GROUP BY a.experiment_id, a.user_id
        ), pre AS (
            SELECT user_id, COUNT(*) AS pre_sessions, SUM(purchased) AS pre_orders, SUM(revenue) AS pre_revenue
            FROM sessions WHERE day >= {pre_lo} AND day < {pre_hi} GROUP BY user_id
        )
        SELECT a.experiment_id, a.user_id, a.variant, a.n_variants, a.first_exposure_day, a.platform,
               u.region, u.city_tier, u.acquisition_channel,
               p.sessions, p.carts, p.checkouts, p.orders, p.revenue,
               CAST(p.orders > 0 AS INT)                          AS converted,
               COALESCE(pre.pre_sessions, 0) AS pre_sessions,
               COALESCE(pre.pre_orders, 0)   AS pre_orders,
               COALESCE(pre.pre_revenue, 0)  AS pre_revenue
        FROM experiment_assignments a
        JOIN post p USING (experiment_id, user_id)
        LEFT JOIN pre USING (user_id)
        LEFT JOIN users u USING (user_id)
    """), "experiment_users")

    lo, hi = C.OBS_WINDOW
    clo, chi = C.CHURN_LABEL_WINDOW
    save(spark.sql(f"""
        WITH obs AS (
            SELECT * FROM sessions WHERE day >= {lo} AND day < {hi}
        ), cats AS (
            SELECT user_id, COUNT(DISTINCT category) AS n_categories,
                   AVG(CAST(category IN ('grocery', 'fashion', 'beauty') AS DOUBLE)) AS staples_view_share
            FROM events WHERE event_name = 'view_item' AND day >= {lo} AND day < {hi}
            GROUP BY user_id
        ), future AS (
            SELECT DISTINCT user_id FROM sessions WHERE day >= {clo} AND day < {chi}
        )
        SELECT o.user_id,
               COUNT(*)                                   AS sessions,
               COUNT(DISTINCT o.day)                      AS active_days,
               {hi} - MAX(o.day)                          AS recency_days,
               MAX(o.day) - MIN(o.day)                    AS active_span_days,
               SUM(o.views)                               AS views,
               AVG(o.views)                               AS views_per_session,
               AVG(o.duration_min)                        AS avg_session_min,
               SUM(o.carted)                              AS carts,
               SUM(o.checkout)                            AS checkouts,
               SUM(o.purchased)                           AS orders,
               ROUND(SUM(o.revenue), 2)                   AS revenue,
               SUM(o.revenue) / NULLIF(SUM(o.purchased), 0)  AS aov,
               SUM(o.carted) / COUNT(*)                   AS cart_rate,
               SUM(o.purchased) / NULLIF(SUM(o.checkout), 0) AS checkout_conv,
               AVG(CASE WHEN o.purchased = 1 THEN CAST(o.discount_pct > 0 AS DOUBLE) END) AS discount_share,
               AVG(CAST(DAYOFWEEK(o.event_date) IN (1, 7) AS DOUBLE)) AS weekend_share,
               FIRST(o.platform)                          AS platform,
               COALESCE(c.n_categories, 0)                AS n_categories,
               c.staples_view_share,
               u.acquisition_channel, u.region, u.city_tier,
               DATEDIFF(DATE '{C.date_str(hi)}', u.signup_date) AS tenure_days,
               CAST(f.user_id IS NULL AS INT)             AS churned
        FROM obs o
        LEFT JOIN cats c USING (user_id)
        LEFT JOIN users u USING (user_id)
        LEFT JOIN future f USING (user_id)
        GROUP BY o.user_id, c.n_categories, c.staples_view_share, u.acquisition_channel, u.region,
                 u.city_tier, u.signup_date, f.user_id
    """), "user_features")

    tables = [r.tableName for r in spark.sql(f"SHOW TABLES IN {C.HIVE_DB}").collect()]
    dq["tables"] = {t: spark.table(t).count() for t in tables}
    dq["etl_seconds"] = round(time.time() - t0, 1)
    C.OUTPUTS.mkdir(exist_ok=True)
    (C.OUTPUTS / "data_quality.json").write_text(json.dumps(dq, indent=2, default=int))
    print(json.dumps(dq, indent=2, default=int))
    spark.stop()
    return dq


if __name__ == "__main__":
    run()
