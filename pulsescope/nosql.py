"""Serve segment profiles and per-user segment + churn scores from MongoDB (document store).

Product services look up `user_profiles` by user_id to personalise; `segment_profiles` backs dashboards.
Skips cleanly when no MongoDB is reachable (start one with `docker compose up -d mongo`).
"""
from __future__ import annotations

import json
import os

import pandas as pd

from . import config as C

MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017")


def run() -> dict:
    from pymongo import MongoClient, UpdateOne
    from pymongo.errors import ServerSelectionTimeoutError

    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=2000)
    try:
        client.admin.command("ping")
    except ServerSelectionTimeoutError:
        print(f"MongoDB not reachable at {MONGO_URI}; skipping NoSQL export")
        return {"skipped": True}
    db = client["pulsescope"]

    seg = pd.read_parquet(C.OUTPUTS / "segments.parquet")
    churn_path = C.OUTPUTS / "churn_scores_test.parquet"
    if churn_path.exists():
        seg = seg.merge(pd.read_parquet(churn_path)[["user_id", "churn_prob"]], on="user_id", how="left")
    docs = json.loads(seg.to_json(orient="records"))
    ops = [UpdateOne({"_id": d["user_id"]}, {"$set": {k: v for k, v in d.items() if v is not None}}, upsert=True)
           for d in docs]
    res = db.user_profiles.bulk_write(ops, ordered=False)
    db.user_profiles.create_index("segment")
    db.user_profiles.create_index([("segment", 1), ("churn_prob", -1)])

    prof = pd.read_csv(C.OUTPUTS / "segment_profiles.csv")
    db.segment_profiles.delete_many({})
    db.segment_profiles.insert_many(json.loads(prof.to_json(orient="records")))

    # Example serving query: highest-risk users of the most valuable segment, via an aggregation pipeline.
    top = list(db.user_profiles.aggregate([
        {"$match": {"churn_prob": {"$exists": True}}},
        {"$group": {"_id": "$segment", "users": {"$sum": 1}, "avg_churn": {"$avg": "$churn_prob"}}},
        {"$sort": {"avg_churn": -1}},
    ]))
    out = {"skipped": False, "upserted": res.upserted_count, "modified": res.modified_count,
           "user_profiles": db.user_profiles.estimated_document_count(),
           "segment_profiles": db.segment_profiles.estimated_document_count(), "churn_by_segment": top}
    (C.OUTPUTS / "nosql.json").write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(out, indent=2, default=str))
    return out
