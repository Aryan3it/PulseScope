"""Spark session factory with a local Derby-backed Hive metastore."""
import os

from pyspark.sql import SparkSession

from . import config as C


def get_spark(app: str = "pulsescope", memory: str = "6g") -> SparkSession:
    C.METASTORE.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("PYSPARK_PYTHON", os.sys.executable)
    spark = (
        SparkSession.builder.appName(app)
        .master("local[*]")
        .config("spark.driver.memory", memory)
        .config("spark.sql.shuffle.partitions", "32")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.warehouse.dir", str(C.WAREHOUSE))
        .config("spark.ui.showConsoleProgress", "false")
        .config("spark.hadoop.javax.jdo.option.ConnectionURL",
                f"jdbc:derby:;databaseName={C.METASTORE / 'metastore_db'};create=true")
        .config("spark.driver.extraJavaOptions", f"-Dderby.system.home={C.METASTORE}")
        .enableHiveSupport()
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark
