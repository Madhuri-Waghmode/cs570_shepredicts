"""
features.py — Build the modeling dataset: history features, fallback values, genre columns, encoding, and the vector.

Every function that raises NotImplementedError is the team's work; keep its name, arguments, and output as the
docstring states. The other functions are complete. build_datasets calls them in order, and D3 calls it again on
other rows. The team may add functions and columns.
"""
import pyspark.sql.functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler

# The four required history features and the two flags, with these names.
HISTORY = ["user_count", "user_avg", "movie_count", "movie_avg"]
FLAGS = ["user_new", "movie_new"]


def add_history(train_df, rows_df=None):
    """
    Purpose: Add user_count, user_avg, movie_count, and movie_avg to each input row.
             Without rows_df, use only training ratings with a strictly earlier Timestamp for that user or movie.
             With rows_df, use all training ratings for that user or movie, and no rating from rows_df.
    In: train_df (DataFrame) with UserID, MovieID, Rating, and Timestamp; rows_df (DataFrame or None) with UserID,
        MovieID, and Timestamp. Timestamp is UTC epoch seconds. When rows_df is supplied, all its timestamps must
        follow the training period. It may have no Rating or label column; neither is needed for its history.
    Out: The input rows (train_df, or rows_df) with the four history columns added. Preserve every input row exactly
         once and keep its existing columns and values. Use count 0 and average null when no allowed history exists.
    """

    raise NotImplementedError("add_history")


def fallback_value(train_df):
    """
    Purpose: The value that replaces a missing history average: the mean Rating of the training rows.
    In: train_df (DataFrame) training rows
    Out: float
    """

    raise NotImplementedError("fallback_value")


def fill_history(df, fallback):
    """
    Purpose: Add the flags user_new and movie_new (1 where the count is 0, else 0) and replace a missing user_avg or
             movie_avg with the fallback value.
    In: df (DataFrame) with the four history features; fallback (float) from fallback_value
    Out: DataFrame with user_new and movie_new added and no null in user_avg or movie_avg
    """

    raise NotImplementedError("fill_history")


def genre_list(train_df):
    """
    Purpose: The genres that occur in the training rows, in alphabetical order.
    In: train_df (DataFrame) training rows with Genres
    Out: list of str
    """

    genres = train_df.select(F.explode(F.split("Genres", r"\|")).alias("genre")).distinct().collect()
    return sorted(r["genre"] for r in genres)


def genre_column(genre):
    """
    Purpose: The column name of one genre indicator, for example g_Sci_Fi for Sci-Fi.
    In: genre (str)
    Out: str
    """

    return "g_" + "".join(c if c.isalnum() else "_" for c in genre)


def add_static(df, genres):
    """
    Purpose: Add gender and genre indicators and the logarithms of the history counts.
    In: df (DataFrame); genres (list of str) from genre_list
    Out: DataFrame with gender_f, the genre columns, user_count_log, and movie_count_log added
    """

    genre_arr = F.split("Genres", r"\|")
    df = df.withColumn("gender_f", (F.col("Gender") == "F").cast("double"))
    for genre in genres:
        df = df.withColumn(genre_column(genre), F.array_contains(genre_arr, genre).cast("double"))

    return (df.withColumn("user_count_log", F.log1p(F.col("user_count").cast("double")))
              .withColumn("movie_count_log", F.log1p(F.col("movie_count").cast("double"))))


def vector_inputs(genres):
    """
    Purpose: The columns that enter the feature vector, in order.
    In: genres (list of str) from genre_list
    Out: list of str
    """

    numeric = ["user_count_log", "user_avg", "user_new", "movie_count_log", "movie_avg", "movie_new", "gender_f"]
    return numeric + [genre_column(g) for g in genres] + ["age_vec", "occupation_vec"]


def encoder_pipeline(genres):
    """
    Purpose: The unfitted Spark Pipeline that one-hot encodes age and occupation and assembles the feature vector.
    In: genres (list of str) from genre_list
    Out: Pipeline
    """

    indexer = StringIndexer(inputCols=["Age", "Occupation"], outputCols=["age_idx", "occupation_idx"],
                            handleInvalid="keep")
    encoder = OneHotEncoder(inputCols=["age_idx", "occupation_idx"], outputCols=["age_vec", "occupation_vec"])
    assembler = VectorAssembler(inputCols=vector_inputs(genres), outputCol="features")

    return Pipeline(stages=[indexer, encoder, assembler])


def check_no_nulls(dfs, columns=None):
    """
    Purpose: Stop with a message naming any columns that still contain nulls.
    In: dfs (list of DataFrames); columns (list of str or None), defaulting to the history features and flags
    Out: none; raises ValueError naming the columns with nulls. Runs one aggregation per DataFrame.
    """

    cols = HISTORY + FLAGS if columns is None else columns
    for df in dfs:
        counts = df.select([F.count(F.when(F.col(c).isNull(), 1)).alias(c) for c in cols]).first().asDict()
        missing = {c: n for c, n in counts.items() if n}
        if missing:
            raise ValueError(f"Null values in feature inputs: {missing}. The vector cannot hold a null.")


def prepare_features(train_raw_df, rows_df, fallback, genres):
    """
    Purpose: Build and check the feature columns before encoding. Add optional feature calculations here so that
             training, test, and production rows use the same definitions.
    In: train_raw_df (DataFrame) history source; rows_df (DataFrame or None), as in add_history;
        fallback (float) and genres (list of str), computed from the training rows
    Out: DataFrame with history features, flags, indicators, and log counts, ready for the encoder
    """

    df = add_static(fill_history(add_history(train_raw_df, rows_df), fallback), genres)
    scalar_inputs = [c for c in vector_inputs(genres) if c not in ("age_vec", "occupation_vec")]
    check_no_nulls([df], list(dict.fromkeys(HISTORY + FLAGS + scalar_inputs)))
    return df


def transform_rows(train_raw_df, rows_df, fitted):
    """
    Purpose: Build features for scoring rows using saved settings. Fits nothing and needs no scoring outcomes.
    In: train_raw_df (DataFrame) history source; rows_df (DataFrame) after the training period, joined to user and
        movie attributes; fitted (dict) with fallback, genres, and encoder from build_datasets or saved files
    Out: DataFrame with the input rows, feature columns, and assembled features vector
    """

    df = prepare_features(train_raw_df, rows_df, fitted["fallback"], fitted["genres"])
    return fitted["encoder"].transform(df)


def build_datasets(train_raw_df, score_raw_dfs):
    """
    Purpose: Build training and scoring datasets, fitting the fallback, genre list, and encoders on training rows only.
    In: train_raw_df (DataFrame) training rows; score_raw_dfs (list of DataFrames) later rows joined to user and movie
        attributes. Scoring rows may have no Rating or label column.
    Out: tuple of (train_df, list of scored DataFrames, fitted). fitted contains fallback, genres, and encoder.
         Use transform_rows with this state during production scoring; build_datasets fits new settings.
    """

    fallback = fallback_value(train_raw_df)
    genres = genre_list(train_raw_df)
    train_df = prepare_features(train_raw_df, None, fallback, genres)

    encoder = encoder_pipeline(genres).fit(train_df)
    fitted = {"fallback": fallback, "genres": genres, "encoder": encoder}
    scored_dfs = [transform_rows(train_raw_df, rows_df, fitted) for rows_df in score_raw_dfs]

    return encoder.transform(train_df), scored_dfs, fitted
