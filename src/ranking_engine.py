"""
ranking_engine.py
------------------
Turns model predictions + raw feature signals into a ranked, explainable
action list. Every row gets one primary action and one or more reason
codes, so a content team can act without having to trust a black box.

Actions (Refresh / Content Opportunity Scoring lane):
  - PRIORITY_REFRESH  : predicted declining, or already declining with a
                         CTR/position gap that suggests a fixable problem.
  - MONITOR           : predicted or observed stable-but-fragile signals.
  - PROTECT           : predicted/observed growing and already a strong
                         performer — don't touch, but keep an eye on it.
  - LOW_PRIORITY      : stable, low-traffic, no clear signal either way.
"""

import numpy as np
import pandas as pd

TOP_PERFORMER_IMPRESSIONS_PCTL = 0.75
LOW_TRAFFIC_IMPRESSIONS_PCTL = 0.25


def _reason_codes(row: pd.Series) -> list[str]:
    codes = []
    if row["position_slope"] > 0.3:
        codes.append("position_declining")
    elif row["position_slope"] < -0.3:
        codes.append("position_improving")

    if row["ctr_gap_vs_expected"] < -0.02:
        codes.append("ctr_below_expected_for_position")
    elif row["ctr_gap_vs_expected"] > 0.02:
        codes.append("ctr_above_expected_for_position")

    if row["impressions_slope"] > 0 and row["clicks_slope"] <= 0:
        codes.append("impressions_up_clicks_flat")

    if row["impressions_sum"] >= row["_impr_p75"]:
        codes.append("high_visibility_page")
    elif row["impressions_sum"] <= row["_impr_p25"]:
        codes.append("low_visibility_page")

    if not codes:
        codes.append("no_strong_signal")
    return codes


def _assign_action(row: pd.Series) -> str:
    predicted = row["predicted_momentum"]
    high_vis = row["impressions_sum"] >= row["_impr_p75"]
    low_vis = row["impressions_sum"] <= row["_impr_p25"]

    if predicted == "declining":
        return "PRIORITY_REFRESH"
    if predicted == "growing" and high_vis:
        return "PROTECT"
    if predicted == "growing":
        return "MONITOR"
    # predicted stable
    if low_vis:
        return "LOW_PRIORITY"
    if row["ctr_gap_vs_expected"] < -0.02:
        return "PRIORITY_REFRESH"
    return "MONITOR"


def build_ranked_recommendations(
    valid_df: pd.DataFrame, model_preds: np.ndarray, model_proba: np.ndarray, classes: list[str]
) -> pd.DataFrame:
    df = valid_df.copy().reset_index(drop=True)
    df["predicted_momentum"] = model_preds

    class_index = {c: i for i, c in enumerate(classes)}
    df["confidence"] = [
        model_proba[i, class_index[pred]] for i, pred in enumerate(model_preds)
    ]

    df["_impr_p75"] = df["impressions_sum"].quantile(TOP_PERFORMER_IMPRESSIONS_PCTL)
    df["_impr_p25"] = df["impressions_sum"].quantile(LOW_TRAFFIC_IMPRESSIONS_PCTL)

    df["action"] = df.apply(_assign_action, axis=1)
    df["reason_codes"] = df.apply(_reason_codes, axis=1).apply(lambda x: ";".join(x))

    action_priority = {"PRIORITY_REFRESH": 0, "PROTECT": 1, "MONITOR": 2, "LOW_PRIORITY": 3}
    df["_action_rank"] = df["action"].map(action_priority)
    df = df.sort_values(
        by=["_action_rank", "confidence"], ascending=[True, False]
    ).reset_index(drop=True)
    df["priority_rank"] = np.arange(1, len(df) + 1)

    out_cols = [
        "priority_rank", "page_id", "action", "predicted_momentum", "confidence",
        "reason_codes", "avg_position_mean", "position_slope", "impressions_sum",
        "ctr_actual", "ctr_gap_vs_expected", "content_type",
    ]
    return df[out_cols]


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/home/claude/Den_Search_Ranking_Discoverability_Capstone")
    from src.features import build_train_valid
    from src.scoring_model import train_and_evaluate

    raw = pd.read_csv(
        "/home/claude/Den_Search_Ranking_Discoverability_Capstone/data/sample_search_data.csv",
        parse_dates=["date"],
    )
    train_df, valid_df = build_train_valid(raw)
    out = train_and_evaluate(train_df, valid_df)
    ranked = build_ranked_recommendations(
        valid_df, out["model_preds"], out["model_proba"], out["results"]["classes"]
    )
    ranked.to_csv(
        "/home/claude/Den_Search_Ranking_Discoverability_Capstone/outputs/ranked_recommendations.csv",
        index=False,
    )
    print(ranked["action"].value_counts())
    print(ranked.head(10).to_string(index=False))
