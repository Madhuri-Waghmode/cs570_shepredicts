"""
evaluate.py — The threshold of the single-feature rule and the accuracy breakdown of the baselines.

Every function that raises NotImplementedError is the team's work; keep its name, arguments, and output as the
docstring states.
"""
import pyspark.sql.functions as F

import cs570_helpers as h


def choose_rule_threshold(train_df, thresholds):
    """
    Purpose: Choose the threshold of the single-feature rule, predict 1 when movie_avg >= t, with the highest accuracy
             on the training rows. On an exact accuracy tie, choose the lower threshold, regardless of list order.
    In: train_df (DataFrame) training rows with movie_avg and label; thresholds (list of float) candidates
    Out: float, the chosen threshold
    """

    raise NotImplementedError("choose_rule_threshold")


def accuracy_breakdown(df, predictions):
    """
    Purpose: Accuracy of one or more 0/1 predictions on the same rows: overall, on rows with both histories, and on
             rows that use a fallback value (user_new or movie_new equal to 1).
    In: df (DataFrame) rows with label, user_new, movie_new; predictions (dict) name to a Spark Column of 0/1 values
    Out: pandas DataFrame indexed by breakdown (overall, both_histories, fallback), with a column rows and one column per
         prediction, accuracies rounded to 4 decimals
    """

    raise NotImplementedError("accuracy_breakdown")
