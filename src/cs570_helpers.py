"""
CS570 helper library: functions reused across the labs and the project.

Keep this file in code/src. A notebook in code/notebooks imports it with

    import sys
    sys.path.append("../src")
    import cs570_helpers as h

A copy placed next to the notebook works with the import line alone.

    show_error             the useful line of a Spark error
    fit_line_and_plot      scatter plot with the least-squares line
    group_means            mean of a target and row count per group
    plot_group_boxes       box-and-whisker plot of a target per group
    spread                 gap between the highest and the lowest group mean
    constant_baseline      predicts the training mean for every test row
    regression_scores      RMSE, MAE and R²
    threshold_table        flagged rows, precision and recall at several thresholds
    classification_report  confusion counts, accuracy, precision, recall, F1, baseline, AUC
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
import pyspark.sql.functions as F
from pyspark.ml.evaluation import BinaryClassificationEvaluator, RegressionEvaluator
from pyspark.ml.functions import vector_to_array

# Spark 4 also logs every query error as a long JSON line; the exception itself still reaches Python
logging.getLogger("DataFrameQueryContextLogger").setLevel(logging.CRITICAL)


# ------------------------------------------------------------------ errors
def show_error(e, needle=None):
    """Print the useful line of a Spark error instead of the whole stack trace.

    Without needle, the first line of the message; with needle, the first line that contains it.
    """

    lines = [line.strip() for line in str(e).splitlines() if line.strip()]
    if needle is not None:
        lines = [line for line in lines if needle in line] or lines

    print(lines[0] if lines else repr(e))


# ------------------------------------------------------------------ looking at the data
def fit_line_and_plot(df, x_col, y_col, size=36, alpha=1.0):
    """Scatter plot of y_col against x_col with the least-squares line; prints its slope and intercept.

    df is a Spark DataFrame. Its rows are collected to the driver, so pass a sample of a large table.
    """

    # the two named columns, renamed x and y, collected to the driver
    rows = df.select(F.col(x_col).alias("x"), F.col(y_col).alias("y")).collect()

    # the rows as NumPy arrays, one value per point
    x = np.array([r["x"] for r in rows], dtype=float)
    y = np.array([r["y"] for r in rows], dtype=float)

    # least-squares straight line: a polynomial of degree 1
    slope, intercept = np.polyfit(x, y, 1)

    plt.scatter(x, y, s=size, alpha=alpha)
    plt.plot(x, slope * x + intercept, color="red")
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(f"{y_col} against {x_col}, with the least-squares line")
    plt.show()

    print(f"slope = {slope:.2f}   intercept = {intercept:.2f}")


def group_means(df, key, target):
    """Mean of the target column and number of rows for each value of the key column, ordered by key.

    Returns a DataFrame with the columns key, mean and n. With a 0/1 target the mean is the rate.
    """

    return (df.groupBy(key)
              .agg(F.avg(target).alias("mean"), F.count("*").alias("n"))
              .orderBy(key))


def plot_group_boxes(df, key, target, title=None, whiskers=(0.05, 0.95)):
    """Box-and-whisker plot of the target for each value of the key column, one evenly spaced box per group.

    The box runs from the 25th to the 75th percentile with a line at the median; the whiskers end at the
    5th and 95th percentiles by default. The percentiles are computed in Spark with percentile_approx,
    so only five numbers per group reach the driver. The row count of each group is printed above its box.
    """

    lo, hi = whiskers
    probs = [lo, 0.25, 0.5, 0.75, hi]

    # one row per group, computed in Spark: the five percentiles as an array q, and the row count n
    rows = (df.groupBy(key)
              .agg(F.percentile_approx(target, probs, 10000).alias("q"), F.count("*").alias("n"))
              .orderBy(key)
              .collect())

    # the precomputed box statistics in the form matplotlib's bxp expects, one dict per box
    stats = [{"label": str(r[key]), "whislo": r["q"][0], "q1": r["q"][1], "med": r["q"][2],
              "q3": r["q"][3], "whishi": r["q"][4], "fliers": []} for r in rows]

    # one figure with one set of axes, wider when there are many groups
    fig, ax = plt.subplots(figsize=(max(6, 1.1 * len(rows)), 4.5))
    ax.bxp(stats, showfliers=False)

    # room above the highest whisker for the row counts
    top = max(s["whishi"] for s in stats)
    bottom = min(s["whislo"] for s in stats)
    ax.set_ylim(bottom - 0.05 * (top - bottom), top + 0.12 * (top - bottom))

    # the row count above each box; the boxes sit at x = 1, 2, 3, ...
    for k, r in enumerate(rows, 1):
        ax.annotate(f"n = {r['n']:,}", (k, top), textcoords="offset points", xytext=(0, 6),
                    ha="center", fontsize=8)

    ax.set_xlabel(f"{key}      box: 25th to 75th percentile, line: median, whiskers: {lo:.0%} to {hi:.0%}")
    ax.set_ylabel(target)
    ax.set_title(title or f"{target} by {key}")
    ax.grid(axis="y")
    plt.xticks(rotation=90 if len(rows) > 12 else 0, fontsize=8)
    plt.tight_layout()
    plt.show()


def spread(means_df, min_n=1):
    """Gap between the highest and the lowest group mean, over groups with at least min_n rows."""

    # big: the groups with enough rows to be credible
    big = means_df.filter(F.col("n") >= min_n)

    # gap: the highest group mean minus the lowest, over those groups
    gap = big.agg(F.max("mean") - F.min("mean")).first()[0]
    return round(gap, 3)


# ------------------------------------------------------------------ regression
def constant_baseline(train, test, label_col):
    """The test rows with a prediction column that holds the training mean of the label."""

    train_mean = train.agg(F.avg(label_col)).first()[0]
    return test.withColumn("prediction", F.lit(train_mean))


def regression_scores(pred, label_col, prediction_col="prediction"):
    """RMSE, MAE and R² of a DataFrame with a label column and a prediction column."""

    result = {}

    # one evaluator per metric; each reads the label and the prediction column
    for metric in ["rmse", "mae", "r2"]:
        evaluator = RegressionEvaluator(labelCol=label_col, predictionCol=prediction_col, metricName=metric)
        result[metric] = round(evaluator.evaluate(pred), 4)

    return result


# ------------------------------------------------------------------ classification
def threshold_table(pred, label_col, thresholds, probability_col="probability"):
    """Flagged rows, precision and recall when a row is flagged at p̂ >= t, for each threshold t.

    pred is the output of a fitted classifier's transform; p̂ is index 1 of the probability vector.
    """

    # p1 = p̂, the probability of the positive class; cached, since every threshold reads it again
    scored = pred.withColumn("p1", vector_to_array(probability_col)[1]).cache()
    positives = scored.filter(F.col(label_col) == 1).count()
    rows = []

    # for each threshold: the rows flagged at p̂ >= t, the true positives among them, precision and recall
    for t in thresholds:
        flagged = scored.filter(F.col("p1") >= t).count()
        tp = scored.filter((F.col("p1") >= t) & (F.col(label_col) == 1)).count()
        precision = tp / flagged if flagged else float("nan")
        recall = tp / positives if positives else float("nan")
        rows.append((t, flagged, precision, recall))
        print(f"t = {t}:  flagged = {flagged:>7,}   precision = {precision:.4f}   recall = {recall:.4f}")

    return rows


def classification_report(pred, label_col, prediction_col="prediction", raw_col="rawPrediction"):
    """Confusion counts and metrics of a binary classifier's predictions, printed and returned as a dict.

    The baseline is the accuracy of always predicting the larger class of these rows.
    """

    # the four cells of the confusion matrix: rows counted per (label, prediction) pair, as a dict
    cells = {(r[label_col], r[prediction_col]): r["count"]
             for r in pred.groupBy(label_col, prediction_col).count().collect()}

    # a missing pair means an empty cell, so its count is 0
    tp, fp = cells.get((1, 1.0), 0), cells.get((0, 1.0), 0)
    fn, tn = cells.get((1, 0.0), 0), cells.get((0, 0.0), 0)
    n = tp + fp + fn + tn

    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if tp else 0.0
    
    report = {
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "accuracy": (tp + tn) / n,
        "baseline": max(tp + fn, fp + tn) / n,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc_roc": BinaryClassificationEvaluator(labelCol=label_col, rawPredictionCol=raw_col,
                                                 metricName="areaUnderROC").evaluate(pred),
        "auc_pr": BinaryClassificationEvaluator(labelCol=label_col, rawPredictionCol=raw_col,
                                                metricName="areaUnderPR").evaluate(pred),
    }

    print(f"TP = {tp:,}   FP = {fp:,}   FN = {fn:,}   TN = {tn:,}")
    print(f"accuracy  = {report['accuracy']:.4f}   baseline = {report['baseline']:.4f}")
    print(f"precision = {precision:.4f}   recall = {recall:.4f}   F1 = {f1:.4f}")
    print(f"AUC-ROC   = {report['auc_roc']:.4f}   AUC-PR = {report['auc_pr']:.4f}")
    return report
