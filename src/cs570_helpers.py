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
    check_history          history features before and after one rating changes
    plot_rate_by_bins      high-rating rate against the mean of a numeric column, per equal-count bin
    count_flags            rows, and rows with each 0/1 flag set, for several DataFrames
    vector_positions       position and name of every entry of an assembled feature vector
    majority_baseline      majority class of the training rows and its accuracy on other rows
    rule_table             accuracy of a one-feature rule at several thresholds
    label_correlations     correlation of each numeric column with the label, strongest first
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
from pyspark.ml.feature import Bucketizer
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


# ------------------------------------------------------------------ building features
def check_history(build, rows, changed_id, new_rating, feature_cols, id_col="row_id",
                  key_cols=("user_id", "movie_id"), rating_col="rating", time_col="timestamp", label_col="label"):
    """Check that history features use only strictly earlier rows, by changing one rating and rebuilding.

    build is the team's function: it takes a DataFrame of rows and returns it with the history columns added,
    each row described from the strictly earlier rows of the same DataFrame. rows is a small table, such as the
    one in the setup-check notebook. The rating of the row changed_id becomes new_rating. A label column, rating >= 4,
    is added when rows has none and is recomputed after the change. Fitted values, such as the fallback prior, stay
    fixed inside build.

    For each key column, the rows that share the changed row's key are listed as earlier, same time, current or
    later, with whether any of feature_cols changed. The check passes when no earlier, same-time or current row
    changed and at least one later row did. Prints the table and returns True or False.
    """

    if label_col not in rows.columns:
        rows = rows.withColumn(label_col, (F.col(rating_col) >= 4).cast("int"))

    changed = rows.withColumn(rating_col, F.when(F.col(id_col) == changed_id, F.lit(new_rating))
                                           .otherwise(F.col(rating_col)))
    changed = changed.withColumn(label_col, (F.col(rating_col) >= 4).cast("int"))

    # before and after, side by side by row id; a feature differs when the two values are not equal or not both null
    before = build(rows).select(id_col, *[F.col(c).alias(f"{c}_before") for c in feature_cols])
    after = build(changed).select(id_col, *[F.col(c).alias(f"{c}_after") for c in feature_cols])
    differs = [~F.col(f"{c}_before").eqNullSafe(F.col(f"{c}_after")) for c in feature_cols]
    any_differs = differs[0]
    for d in differs[1:]:
        any_differs = any_differs | d

    both = before.join(after, id_col).withColumn("feature_changed", any_differs)
    target = rows.filter(F.col(id_col) == changed_id).first()
    passed = True

    # for each key: the rows with the changed row's key, placed in time relative to the changed row
    for key in key_cols:
        position = (F.when(F.col(id_col) == changed_id, "current")
                     .when(F.col(time_col) < target[time_col], "earlier")
                     .when(F.col(time_col) == target[time_col], "same time")
                     .otherwise("later"))
        table = (rows.filter(F.col(key) == target[key])
                     .withColumn("position", position)
                     .join(both.select(id_col, "feature_changed"), id_col)
                     .select(id_col, key, time_col, "position", "feature_changed")
                     .orderBy(time_col, id_col)
                     .collect())

        wrong = [r for r in table if r["position"] != "later" and r["feature_changed"]]
        later_changed = [r for r in table if r["position"] == "later" and r["feature_changed"]]
        key_passed = not wrong and bool(later_changed)
        passed = passed and key_passed

        print(f"{key} = {target[key]}")
        for r in table:
            print(f"  {r[id_col]:<14} {time_col} = {r[time_col]:<8} {r['position']:<10} "
                  f"feature changed: {r['feature_changed']}")

        print(f"  {'pass' if key_passed else 'FAIL'}: earlier, same-time and current rows unchanged, a later row changed")

    return passed


def plot_rate_by_bins(df, col, label_col="label", n_bins=10, title=None, path=None):
    """High-rating rate against the mean of a numeric column, for bins that hold about the same number of rows.

    The bin edges are approximate quantiles of col, so a long-tailed count gets narrow bins where most rows lie.
    A straight pattern means a linear model can use the column as it is; a curve or a plateau suggests another
    representation. Comparing the plot of a count with the plot of its logarithm shows which is closer to straight.
    Rows with a missing value of col are left out. The row counts of the bins are printed, not plotted, since the
    bins hold about the same number of rows. Prints the bins and returns them as rows of (mean of col, rate, n).
    With path, the figure is saved there
    instead of shown, which suits a script.
    """

    # quantile edges; repeated edges of a column with many equal values collapse into one
    edges = sorted(set(df.approxQuantile(col, [k / n_bins for k in range(1, n_bins)], 0.001)))
    splits = [-float("inf")] + edges + [float("inf")]
    bucketizer = Bucketizer(splits=splits, inputCol=col, outputCol="bin", handleInvalid="skip")
    binned = bucketizer.transform(df.withColumn(col, F.col(col).cast("double")))

    rows = (binned.groupBy("bin")
                  .agg(F.avg(col).alias("x"), F.avg(label_col).alias("rate"), F.count("*").alias("n"))
                  .orderBy("bin")
                  .collect())

    x = [r["x"] for r in rows]
    rate = [r["rate"] for r in rows]
    plt.figure(figsize=(7, 4.5))
    plt.plot(x, rate, marker="o")
    plt.ylim(0, 1)
    plt.xlabel(f"mean of {col} in the bin")
    plt.ylabel(f"share of {label_col} = 1")
    plt.title(title or f"{label_col} rate by bins of {col}")
    plt.grid(axis="y")
    plt.tight_layout()
    if path is None:
        plt.show()
    else:
        plt.savefig(path)
        plt.close()

    for r in rows:
        print(f"{col} mean = {r['x']:>10.3f}   rate = {r['rate']:.4f}   n = {r['n']:>8,}")

    return [(r["x"], r["rate"], r["n"]) for r in rows]


def count_flags(frames, flag_cols):
    """Rows, and rows with each 0/1 flag column equal to 1, for each DataFrame in frames.

    frames is a dict from a name, such as "train" or "test", to a DataFrame. Prints one line per DataFrame and
    flag with the count and the share, and returns them as rows of (name, flag, count, share).
    """

    result = []

    for name, df in frames.items():
        sums = df.agg(F.count("*").alias("n"), *[F.sum(c).alias(c) for c in flag_cols]).first()
        print(f"{name}: {sums['n']:,} rows")
        for c in flag_cols:
            count = int(sums[c] or 0)
            share = count / sums["n"] if sums["n"] else float("nan")
            result.append((name, c, count, share))
            print(f"  {c:<20} {count:>9,}   share = {share:.4f}")

    return result


def vector_positions(df, features_col="features"):
    """Position and name of every entry of an assembled feature vector, read from the column's metadata.

    df is the output of a fitted VectorAssembler, or of a pipeline that ends with one. A one-hot block is
    named after its column and the category labels of the fitted StringIndexer, so occupation_vec_4 is the
    entry for occupation 4. Prints one line per position and returns the rows as (position, name).
    """

    groups = df.schema[features_col].metadata["ml_attr"]["attrs"]

    # the metadata keeps numeric, binary and nominal entries in separate lists; together they cover every position
    rows = sorted((a["idx"], a["name"]) for kind in groups.values() for a in kind)
    for idx, name in rows:
        print(f"{idx:>4}  {name}")

    return rows


# ------------------------------------------------------------------ baselines
def majority_baseline(train, test, label_col="label"):
    """The class that is more common in the training rows, and the accuracy of always predicting it on the test rows.

    The class is chosen once on train and kept fixed. This differs from the baseline printed by
    classification_report, which is the larger class of the evaluated rows themselves.
    """

    majority = (train.groupBy(label_col).count()
                     .orderBy(F.desc("count"), F.asc(label_col))
                     .first()[label_col])

    accuracy = test.agg(F.avg((F.col(label_col) == majority).cast("double"))).first()[0]
    print(f"majority class on the training rows = {majority}   test accuracy = {accuracy:.4f}")
    return majority, accuracy


def rule_table(df, score_col, thresholds, label_col="label"):
    """Accuracy of the rule that predicts 1 when score_col >= t, for each threshold t.

    Run it on the training rows to choose a threshold, then on the test rows with that one threshold.
    Prints one line per threshold and returns the rows as (t, predicted 1, accuracy).
    """

    rows = []

    for t in thresholds:
        pred = (F.col(score_col) >= t).cast("int")
        r = df.agg(F.sum(pred).alias("flagged"), F.avg((pred == F.col(label_col)).cast("double")).alias("acc")).first()
        rows.append((t, r["flagged"], r["acc"]))
        print(f"t = {t}:  predicted 1 = {r['flagged']:>9,}   accuracy = {r['acc']:.4f}")

    return rows


def label_correlations(df, cols, label_col="label"):
    """Correlation of each numeric column with the 0/1 label, ordered by its absolute value, strongest first.

    A correlation measures a straight-line relationship only; a column with a curved pattern can score low.
    It suits measurements and 0/1 indicators, not codes such as an occupation number.
    Prints one line per column and returns the rows as (column, correlation).
    """

    values = df.agg(*[F.corr(c, label_col).alias(c) for c in cols]).first()
    rows = sorted(((c, values[c]) for c in cols), key=lambda r: -abs(r[1] or 0))
    for c, r in rows:
        print(f"{c:<24} {r:+.4f}" if r is not None else f"{c:<24} undefined")

    return rows


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
