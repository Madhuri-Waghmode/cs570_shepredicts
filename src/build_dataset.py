"""
build_dataset.py — Build the modeling datasets, check them, and write their summary files.

From the repository root:
  python src/build_dataset.py
Prints the row counts, the null check, and the rows without history, and writes reports/dataset_summary.csv,
reports/dataset_sample.csv, and reports/vector_positions.csv. Complete as supplied; the work is in features.py.
"""
import pandas as pd
import pyspark.sql.functions as F

import config
import cs570_helpers as h
import data
import features

SUMMARY_COLUMNS = ["label"] + features.HISTORY + features.FLAGS


def summary(df, name):
    """
    Purpose: Summarize the label, the history features, and the flags of one dataset: rows, non-missing values, mean, minimum, and maximum.
    In: df (DataFrame) a modeling dataset; name (str) train or test
    Out: pandas DataFrame with one row per column
    """

    aggs = [F.count("*").alias("rows")]
    for c in SUMMARY_COLUMNS:
        aggs += [F.count(c).alias(f"{c}__n"), F.avg(c).alias(f"{c}__mean"), F.min(c).alias(f"{c}__min"),
                 F.max(c).alias(f"{c}__max")]

    r = df.agg(*aggs).first()

    # one line per column, rounded so that two correct builds agree exactly
    return pd.DataFrame([{"dataset": name, "column": c, "rows": r["rows"], "non_missing": r[f"{c}__n"],
                          "mean": round(r[f"{c}__mean"], 4), "min": round(float(r[f"{c}__min"]), 4),
                          "max": round(float(r[f"{c}__max"]), 4)} for c in SUMMARY_COLUMNS])


def main():
    """
    Purpose: Build the training and test datasets, run the checks, and write the summary files.
    In: none
    Out: none
    """

    spark = config.spark_session("build_dataset")
    config.REPORTS.mkdir(exist_ok=True)

    # the three files, repaired and joined, then split at 2000-12-01 00:00 UTC
    train_raw_df, test_raw_df = data.split(data.load_ratings(spark))
    n_train, n_test = train_raw_df.count(), test_raw_df.count()
    print(f"Training rows {n_train:,}, test rows {n_test:,}")

    # history, fallback values, static columns, and the vector, all fitted on the training rows
    train_df, (test_df,), fitted = features.build_datasets(train_raw_df, [test_raw_df])
    train_df, test_df = train_df.cache(), test_df.cache()
    print(f"Fallback value {fitted['fallback']:.4f}, {len(fitted['genres'])} genres")

    # check 1: building the features keeps every row
    same_rows = train_df.count() == n_train and test_df.count() == n_test
    print(f"Check, rows unchanged: {same_rows}")

    # check 2: no null in a column that enters the vector
    inputs = [c for c in features.vector_inputs(fitted["genres"]) if not c.endswith("_vec")]
    features.check_no_nulls([train_df, test_df], inputs)
    print("Check, no nulls in the vector inputs: True")
    if not same_rows:
        raise SystemExit("A check failed. The reports were not updated; fix the features and rerun.")

    # rows without history, on each side
    h.count_flags({"train": train_df, "test": test_df}, features.FLAGS)

    # the files the instructor compares across teams
    pd.concat([summary(train_df, "train"), summary(test_df, "test")]).to_csv(
        config.REPORTS / "dataset_summary.csv", index=False)
    sample_cols = ["UserID", "MovieID", "Timestamp", "label"] + features.HISTORY + features.FLAGS
    sample = pd.concat([df.filter(F.col("UserID").isin(config.SAMPLE_USERS)).select(sample_cols).toPandas()
                        .assign(dataset=name) for name, df in (("train", train_df), ("test", test_df))])
    sample.sort_values(["dataset", "UserID", "Timestamp", "MovieID"]).round(4).to_csv(
        config.REPORTS / "dataset_sample.csv", index=False)
    positions = h.vector_positions(train_df)
    pd.DataFrame(positions, columns=["position", "name"]).to_csv(config.REPORTS / "vector_positions.csv", index=False)
    print(f"Wrote dataset_summary.csv, dataset_sample.csv ({len(sample)} rows) and vector_positions.csv "
          f"({len(positions)} positions)")
    spark.stop()


if __name__ == "__main__":
    main()
