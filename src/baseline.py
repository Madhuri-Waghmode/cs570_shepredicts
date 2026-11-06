"""
baseline.py — Accuracy of the two baselines on the test rows: the training majority class and the single-feature rule.

From the repository root:
  python src/baseline.py
Writes reports/baseline_accuracy.csv. Complete as supplied; the work is in evaluate.py.
"""
import pyspark.sql.functions as F

import config
import cs570_helpers as h
import data
import evaluate
import features

THRESHOLDS = [3.0, 3.25, 3.5, 3.75, 4.0]


def main():
    """
    Purpose: Fix both baselines on the training rows and write their accuracy breakdown on the test rows.
    In: none
    Out: none
    """

    spark = config.spark_session("baseline")
    config.REPORTS.mkdir(exist_ok=True)
    train_raw_df, test_raw_df = data.split(data.load_ratings(spark))
    train_df, (test_df,), _ = features.build_datasets(train_raw_df, [test_raw_df])

    # both baselines are fixed on the training rows: the more common class, and the rule's threshold
    majority, _ = h.majority_baseline(train_df, test_df)
    threshold = evaluate.choose_rule_threshold(train_df, THRESHOLDS)
    print(f"Rule: predict a high rating when movie_avg >= {threshold}")

    predictions = {"majority": F.lit(majority), "rule": (F.col("movie_avg") >= threshold).cast("int")}
    table = evaluate.accuracy_breakdown(test_df, predictions).rename_axis("breakdown").reset_index()
    table["rule_threshold"] = threshold
    table.to_csv(config.REPORTS / "baseline_accuracy.csv", index=False)
    print(table.to_string(index=False))
    spark.stop()


if __name__ == "__main__":
    main()
