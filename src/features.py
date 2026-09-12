"""
features.py
------------
Builds page-level features and forward-looking labels from the daily
warehouse using DuckDB, following the workflow taught in the program's
Starter Notebook 03 (aggregate with DuckDB, model in sklearn).

VALIDATION / LEAKAGE DESIGN (out-of-time split)
------------------------------------------------
Two independent (feature_window -> label_window) cutoffs are built:

  TRAIN  : features from day  31-90   -> label from day  91-150
  VALID  : features from day  61-120  -> label from day 121-180

The validation cutoff starts strictly *after* the training cutoff's
feature window ends, so the model is always evaluated on a period the
training process could not have seen — a standard out-of-time check for
time-series-style panel data. `true_regime` (the synthetic ground truth)
is deliberately excluded from the feature set; it is only reattached in
the evaluation notebook to sanity-check the simulation, never passed to
the model.
"""

from dataclasses import dataclass

import duckdb
import pandas as pd


@dataclass
class WindowSpec:
    name: str
    feat_start: int
    feat_end: int
    label_start: int
    label_end: int


TRAIN_WINDOW = WindowSpec("train", 31, 90, 91, 150)
VALID_WINDOW = WindowSpec("valid", 61, 120, 121, 180)

MOMENTUM_GROWTH_THRESHOLD = 0.15   # +15% clicks -> "growing"
MOMENTUM_DECLINE_THRESHOLD = -0.15  # -15% clicks -> "declining"

FEATURE_SQL = """
WITH indexed AS (
    SELECT
        *,
        DATE_DIFF('day', MIN(date) OVER (), date) AS day_idx
    FROM raw
),
feat AS (
    SELECT
        page_id,
        ANY_VALUE(content_type)  AS content_type,
        ANY_VALUE(word_count)    AS word_count,
        ANY_VALUE(page_age_days) AS page_age_days,
        AVG(avg_position)        AS avg_position_mean,
        REGR_SLOPE(avg_position, day_idx) AS position_slope,
        SUM(impressions)         AS impressions_sum,
        SUM(clicks)              AS clicks_sum,
        SUM(clicks) * 1.0 / NULLIF(SUM(impressions), 0) AS ctr_actual,
        REGR_SLOPE(impressions, day_idx) AS impressions_slope,
        REGR_SLOPE(clicks, day_idx)      AS clicks_slope
    FROM indexed
    WHERE day_idx BETWEEN {feat_start} AND {feat_end}
    GROUP BY page_id
),
label AS (
    SELECT
        page_id,
        SUM(clicks) AS clicks_sum_future
    FROM indexed
    WHERE day_idx BETWEEN {label_start} AND {label_end}
    GROUP BY page_id
)
SELECT
    f.*,
    l.clicks_sum_future,
    (l.clicks_sum_future - f.clicks_sum) * 1.0 / NULLIF(f.clicks_sum, 0) AS click_growth_rate
FROM feat f
JOIN label l USING (page_id)
"""


def _expected_ctr_vec(position: pd.Series) -> pd.Series:
    from src.data_gen import _expected_ctr
    return position.apply(_expected_ctr)


def build_window(raw: pd.DataFrame, window: WindowSpec) -> pd.DataFrame:
    """Run the DuckDB aggregation for one (feature_window -> label_window)
    pair and attach the derived, human-readable features used by the
    ranking engine's reason codes."""
    con = duckdb.connect()
    con.register("raw", raw)
    sql = FEATURE_SQL.format(
        feat_start=window.feat_start, feat_end=window.feat_end,
        label_start=window.label_start, label_end=window.label_end,
    )
    df = con.execute(sql).df()
    con.close()

    df["expected_ctr"] = _expected_ctr_vec(df["avg_position_mean"])
    df["ctr_gap_vs_expected"] = df["ctr_actual"] - df["expected_ctr"]

    def classify(rate: float) -> str:
        if pd.isna(rate):
            return "stable"
        if rate >= MOMENTUM_GROWTH_THRESHOLD:
            return "growing"
        if rate <= MOMENTUM_DECLINE_THRESHOLD:
            return "declining"
        return "stable"

    df["momentum_class"] = df["click_growth_rate"].apply(classify)
    df["window"] = window.name
    return df


def build_train_valid(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = build_window(raw, TRAIN_WINDOW)
    valid_df = build_window(raw, VALID_WINDOW)
    return train_df, valid_df


if __name__ == "__main__":
    raw = pd.read_csv(
        "/home/claude/Den_Search_Ranking_Discoverability_Capstone/data/sample_search_data.csv",
        parse_dates=["date"],
    )
    train_df, valid_df = build_train_valid(raw)
    print("Train window:", train_df.shape, train_df["momentum_class"].value_counts().to_dict())
    print("Valid window:", valid_df.shape, valid_df["momentum_class"].value_counts().to_dict())
