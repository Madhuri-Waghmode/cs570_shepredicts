"""
data.py — Read the three movie files, apply the team's D1 repairs, join them, add the target, and split by time.

Complete add_label with the team's D1 target expression. Add any D1 repairs to apply_repairs.
The other functions are complete.
"""
import pyspark.sql.functions as F

import config

RATINGS_SCHEMA = "UserID INT, MovieID INT, Rating INT, Timestamp LONG"
USERS_SCHEMA = "UserID INT, Gender STRING, Age INT, Occupation INT, Zip STRING"
MOVIES_SCHEMA = "MovieID INT, Title STRING, Genres STRING"


def read_dat(spark, name, schema, encoding="UTF-8"):
    """
    Purpose: Read one '::'-separated file with no header and no quoting.
    In: spark (SparkSession); name (str) file name in config.DEV_DIR; schema (str) DDL schema; encoding (str)
    Out: Spark DataFrame
    """

    path = (config.DEV_DIR / name).as_posix()
    return (spark.read.option("sep", "::").option("encoding", encoding).option("quote", "\u0000")
            .schema(schema).csv(path))


def load_tables(spark):
    """
    Purpose: Read the ratings, users, and movies files; the movies file is Latin-1.
    In: spark (SparkSession)
    Out: tuple of (ratings_df, users_df, movies_df)
    """

    ratings_df = read_dat(spark, "ratings_dev.dat", RATINGS_SCHEMA)
    users_df = read_dat(spark, "users_dev.dat", USERS_SCHEMA)
    movies_df = read_dat(spark, "movies_dev.dat", MOVIES_SCHEMA, encoding="ISO-8859-1")

    return ratings_df, users_df, movies_df


def apply_repairs(ratings_df, users_df, movies_df):
    """
    Purpose: Apply the repairs the team decided on in D1 Part 3. Edit this function; as supplied it changes nothing.
    In: ratings_df, users_df, movies_df (DataFrames) as read by load_tables
    Out: tuple of (ratings_df, users_df, movies_df) after the repairs
    """

    # D1 found no duplicates, orphan keys, nulls, or out-of-domain values, so nothing is changed.
    return ratings_df, users_df, movies_df


def join_tables(ratings_df, users_df, movies_df):
    """
    Purpose: Join the repaired ratings to their user and movie attributes.
    In: ratings_df, users_df, movies_df (DataFrames) after the repairs
    Out: DataFrame with one row per rating
    """

    return (ratings_df.join(users_df, "UserID", "inner")
            .join(movies_df, "MovieID", "inner")
            .drop("Zip"))


def add_label(df):
    """
    Purpose: Create the fixed target using the team's D1 expression, after repairs and joins and before splitting.
    In: df (DataFrame) with valid Rating values from 1 to 5
    Out: DataFrame with an integer label column: 1 for Rating 4 or 5, otherwise 0. Preserve every input row and
         all other columns and values.
    """

    return df.withColumn("label", F.when(F.col("Rating") >= 4, 1).otherwise(0))


def load_ratings(spark):
    """
    Purpose: Read, repair, and join the three files, then call the team's add_label function.
    In: spark (SparkSession)
    Out: DataFrame with one row per rating and the label
    """

    return add_label(join_tables(*apply_repairs(*load_tables(spark))))


def split(df):
    """
    Purpose: Split the ratings at the fixed cutoff, 2000-12-01 00:00 UTC.
    In: df (DataFrame) from load_ratings
    Out: tuple of (train_raw_df, test_raw_df)
    """

    return df.filter(F.col("Timestamp") < config.CUTOFF), df.filter(F.col("Timestamp") >= config.CUTOFF)
