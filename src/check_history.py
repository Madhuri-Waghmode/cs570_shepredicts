"""
check_history.py — Check training histories and the scoring branch on small supplied tables.

From the repository root:
  python src/check_history.py
Changes the rating of row b from 4 to 1 and rebuilds the history with features.add_history. Earlier, same-second,
and current rows must keep their history; a later row of the same user and of the same movie must change.
Checks add_history(train, rows) against expected training-only histories, with scoring outcomes changed and removed.
Checks exact training histories before and after the rating change, including all input rows and columns.
Writes reports/history_check.csv and stops with an error if the check fails. Complete as supplied.
"""
import math

import pandas as pd
import pyspark.sql.functions as F

import config
import cs570_helpers as h
import features

# user 1 and movie 10 each have an earlier row, a row in the same second as row b, and a later row
TABLE = [("a_user", 1, 11, 3, 100), ("a_movie", 2, 10, 3, 100), ("b", 1, 10, 4, 200),
         ("same_user", 1, 12, 2, 200), ("same_movie", 3, 10, 2, 200),
         ("later_user", 1, 13, 5, 300), ("later_movie", 4, 10, 5, 300)]

# All scoring timestamps follow the training rows, including a later rating of the same user and movie.
SCORE_TABLE = [("known", 1, 10, 5, 400), ("new_user", 99, 10, 2, 500),
               ("new_movie", 1, 99, 4, 500), ("both_new", 99, 99, 1, 500)]


def check_expected_history(train, rows, expected, name, build):
    """
    Purpose: Check exact history values and preserve every input row and its original columns.
    In: train (DataFrame); rows (DataFrame or None); expected (dict) row_id to four history values;
        name (str) printed description; build (callable) with the add_history interface
    Out: bool; prints pass or FAIL for this example
    """

    inputs = train if rows is None else rows
    try:
        originals = {r["row_id"]: r.asDict() for r in inputs.collect()}
        result = build(train) if rows is None else build(train, rows)
        output = result.select(*inputs.columns, *features.HISTORY).collect()
        actual = {r["row_id"]: r for r in output}
        passed = len(output) == len(originals) and actual.keys() == originals.keys() == expected.keys()
        if passed:
            passed = all(
                all(actual[key][c] == value for c, value in original.items()) and
                all(value is None if target is None else value is not None and
                    math.isclose(float(value), float(target), rel_tol=0, abs_tol=1e-10)
                    for value, target in zip((actual[key][c] for c in features.HISTORY), expected[key]))
                for key, original in originals.items()
            )
    except Exception as error:
        print(f"{name}: FAIL ({type(error).__name__}: {error})")
        return False

    print(f"{name}: {'pass' if passed else 'FAIL'} (exact histories; all input rows and columns preserved)")
    return passed


def check_training_history(spark, build=features.add_history):
    """
    Purpose: Check exact training histories and the effect of changing one rating.
    In: spark (SparkSession); build (callable) with the add_history interface
    Out: bool; both exact examples and the strictly-earlier mutation check must pass
    """

    train = spark.createDataFrame(TABLE, ["row_id", "UserID", "MovieID", "Rating", "Timestamp"])
    train = train.withColumn("label", (F.col("Rating") >= 4).cast("int"))
    expected = {
        "a_user": (0, None, 0, None), "a_movie": (0, None, 0, None),
        "b": (1, 3.0, 1, 3.0), "same_user": (1, 3.0, 0, None),
        "same_movie": (0, None, 1, 3.0), "later_user": (3, 3.0, 0, None),
        "later_movie": (0, None, 3, 3.0),
    }
    changed = train.withColumn("Rating", F.when(F.col("row_id") == "b", 1).otherwise(F.col("Rating")))
    changed = changed.withColumn("label", (F.col("Rating") >= 4).cast("int"))
    expected_changed = dict(expected, later_user=(3, 2.0, 0, None), later_movie=(0, None, 3, 2.0))
    original_passed = check_expected_history(train, None, expected, "Training history, original", build)
    changed_passed = check_expected_history(changed, None, expected_changed, "Training history, changed", build)
    mutation_passed = h.check_history(build, train, "b", 1, features.HISTORY,
                                      key_cols=("UserID", "MovieID"), rating_col="Rating", time_col="Timestamp")
    return original_passed and changed_passed and mutation_passed


def check_scoring_history(spark, build=features.add_history):
    """
    Purpose: Verify exact scoring histories and row preservation when scoring outcomes change or are absent.
    In: spark (SparkSession); build (callable) with the add_history(train, rows) interface
    Out: bool; prints the result of each supplied case
    """

    columns = ["row_id", "UserID", "MovieID", "Rating", "Timestamp"]
    train = spark.createDataFrame(TABLE, columns)
    rows = spark.createDataFrame(SCORE_TABLE, columns).withColumn("label", (F.col("Rating") >= 4).cast("int"))
    changed = rows.withColumn("Rating", 6 - F.col("Rating"))
    changed = changed.withColumn("label", (F.col("Rating") >= 4).cast("int"))
    keys = ["row_id", "UserID", "MovieID", "Timestamp"]
    expected = {
        ("known", 1, 10, 400): (4, 3.5, 4, 3.5),
        ("new_user", 99, 10, 500): (0, None, 4, 3.5),
        ("new_movie", 1, 99, 500): (4, 3.5, 0, None),
        ("both_new", 99, 99, 500): (0, None, 0, None),
    }
    history = ["user_count", "user_avg", "movie_count", "movie_avg"]
    passed = True
    cases = {"original outcomes": rows, "changed outcomes": changed,
             "outcomes absent": rows.drop("Rating", "label")}
    for name, score_rows in cases.items():
        try:
            output = build(train, score_rows).select(*keys, *history).collect()
            actual = {tuple(r[c] for c in keys): tuple(r[c] for c in history) for r in output}
            case_passed = len(output) == len(expected) and actual.keys() == expected.keys()
            if case_passed:
                case_passed = all(
                    (value is None if target is None else value is not None and
                     math.isclose(float(value), float(target), rel_tol=0, abs_tol=1e-10))
                    for key, targets in expected.items() for value, target in zip(actual[key], targets)
                )
        except Exception as error:
            print(f"Scoring history, {name}: FAIL ({type(error).__name__}: {error})")
            case_passed = False
        else:
            print(f"Scoring history, {name}: {'pass' if case_passed else 'FAIL'} "
                  "(four rows preserved; expected training-only counts and averages)")
        passed = passed and case_passed

    return passed


def main():
    """
    Purpose: Run the training and scoring history checks and write their results.
    In: none
    Out: none
    """

    spark = config.spark_session("check_history")
    config.REPORTS.mkdir(exist_ok=True)
    train_passed = check_training_history(spark)
    score_passed = check_scoring_history(spark)
    pd.DataFrame([{"check": "strictly earlier history", "changed_row": "b", "passed": train_passed},
                  {"check": "scoring history independent of outcomes", "changed_row": "all scoring rows",
                   "passed": score_passed}]).to_csv(
        config.REPORTS / "history_check.csv", index=False)
    spark.stop()
    if not train_passed or not score_passed:
        raise SystemExit("A history check failed. Check the printed training and scoring cases, including "
                         "same-second exclusion, training-only scoring histories, and row preservation.")


if __name__ == "__main__":
    main()
