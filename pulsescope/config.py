"""Central configuration: paths, calendar, and the planted scenario timeline."""
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"
RAW_EVENTS = RAW / "events"
WAREHOUSE = DATA / "warehouse"          # Spark/Hive managed tables (Parquet)
METASTORE = DATA / "metastore"          # Derby-backed Hive metastore
LAKE = WAREHOUSE / "pulsescope.db"
OUTPUTS = ROOT / "outputs"
SQL_DIR = Path(__file__).resolve().parent / "sql"

HIVE_DB = "pulsescope"

START_DATE = date(2026, 4, 1)
N_DAYS = 91

# Scenario timeline (day offsets from START_DATE).
CAMPAIGN_DAY = 30                         # one-day email campaign (a *benign* anomaly)
IOS_RELEASE_DAY = 45                      # benign iOS release (red herring for RCA)
INCIDENT_START, INCIDENT_END = 60, 67     # Android 5.2.0 checkout bug, [start, end)
EXP_A = dict(id="one_click_checkout", start=70, end=N_DAYS)
EXP_B = dict(id="reco_widget_v2", start=75, end=N_DAYS)

# Analysis windows.
OBS_WINDOW = (0, 60)                      # features for segmentation / churn
CHURN_LABEL_WINDOW = (60, N_DAYS)         # churned = no session in this window
CUPED_PRE_WINDOW = (40, 70)               # pre-experiment covariate window


def day_to_date(d: int) -> date:
    return START_DATE + timedelta(days=int(d))


def date_str(d: int) -> str:
    return day_to_date(d).isoformat()


for p in (DATA, RAW, OUTPUTS):
    p.mkdir(parents=True, exist_ok=True)
