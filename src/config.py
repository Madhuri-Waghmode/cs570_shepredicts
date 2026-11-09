"""
config.py — Paths, the fixed split, and the Spark session that every script of the movie project uses.

Paths are relative to the repository root. Complete as supplied; the values are the same for every team.
"""
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEV_DIR = REPO_ROOT / "data" / "movie" / "dev"
REPORTS = REPO_ROOT / "reports"
FIGURES = REPO_ROOT / "figures"

# Fixed for every team: training rows before the cutoff, test rows (December 2000) from the cutoff on.
CUTOFF = int(datetime(2000, 12, 1, tzinfo=timezone.utc).timestamp())   # 975628800 seconds
SEED = 42

# The users whose rows are written to reports/dataset_sample.csv: many earlier ratings, few, and none before the cutoff.
SAMPLE_USERS = [673, 745, 20]


def spark_session(app="cs570-project"):
    """
    Purpose: Start the Spark session of a script: all local cores, 2 GB of memory, timestamps in UTC.
    In: app (str) the application name shown by Spark
    Out: SparkSession
    """

    from pyspark.sql import SparkSession

    os.environ["PYSPARK_PYTHON"] = sys.executable   # the Python workers use this environment
    spark = (SparkSession.builder
             .appName(app)
             .master("local[*]")
             .config("spark.driver.memory", "2g")
             .config("spark.sql.session.timeZone", "UTC")
             .config("spark.sql.execution.arrow.pyspark.enabled", "false")
             .getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    logging.getLogger("py4j").setLevel(logging.ERROR)

    return spark
