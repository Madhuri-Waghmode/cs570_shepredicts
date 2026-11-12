
"""Evaluation helpers for D2."""

import pandas as pd
import pyspark.sql.functions as F

import cs570_helpers as h


def choose_rule_threshold(train_df, thresholds):
    """
    Choose the threshold with the highest training accuracy.
    On ties, choose the lower threshold.
    """
    if not thresholds:
        raise ValueError("thresholds must not be empty")

    best_threshold = None
    best_accuracy = -1.0

    for t in sorted(thresholds):
        accuracy = train_df.select(
            F.avg(
                (F.col("movie_avg") >= F.lit(t))
                .cast("int")
                .eqNullSafe(F.col("label").cast("int"))
                .cast("double")
            ).alias("accuracy")
        ).first()["accuracy"]

        if accuracy is not None and accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = t

    if best_threshold is None:
        raise ValueError("No valid training rows to evaluate")

    return float(best_threshold)


def accuracy_breakdown(df, predictions):
    """
    Calculate accuracy overall, for both histories,
    and for fallback rows.
    """
    both = (F.col("user_new") == 0) & (F.col("movie_new") == 0)
    fallback = (F.col("user_new") == 1) | (F.col("movie_new") == 1)

    groups = {
        "overall": df,
        "both_histories": df.filter(both),
        "fallback": df.filter(fallback),
    }

    results = []

    for group_name, subset in groups.items():
        expressions = [F.count("*").alias("rows")]

        for name, prediction in predictions.items():
            correct = (
                prediction.cast("int")
                .eqNullSafe(F.col("label").cast("int"))
            )
            expressions.append(
                F.avg(correct.cast("double")).alias(name)
            )

        values = subset.agg(*expressions).first().asDict()
        values["breakdown"] = group_name

        for name in predictions:
            if values[name] is not None:
                values[name] = round(values[name], 4)

        results.append(values)

    return pd.DataFrame(results).set_index("breakdown").loc[
        ["overall", "both_histories", "fallback"]
    ]
