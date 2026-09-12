"""
scoring_model.py
-----------------
Trains a momentum-class model (growing / stable / declining) on the TRAIN
window and evaluates it, alongside a simple rule-based baseline, on the
strictly-later VALID window (see features.py for the leakage-safe split
design).
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

NUMERIC_FEATURES = [
    "avg_position_mean", "position_slope", "impressions_sum",
    "impressions_slope", "clicks_slope", "ctr_actual",
    "ctr_gap_vs_expected", "word_count", "page_age_days",
]
CATEGORICAL_FEATURES = ["content_type"]
TARGET = "momentum_class"


def baseline_predict(df: pd.DataFrame) -> pd.Series:
    """Rule-based baseline every model must beat: classify purely from the
    sign/magnitude of the position slope (a single, already-available
    signal), with no learned weights at all."""
    def rule(slope: float) -> str:
        if slope <= -0.5:      # position improving (lower number) fast
            return "growing"
        if slope >= 0.5:       # position worsening fast
            return "declining"
        return "stable"
    return df["position_slope"].apply(rule)


def build_pipeline() -> Pipeline:
    pre = ColumnTransformer([
        ("num", "passthrough", NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])
    clf = RandomForestClassifier(
        n_estimators=300, max_depth=6, min_samples_leaf=5,
        random_state=42, class_weight="balanced",
    )
    return Pipeline([("pre", pre), ("clf", clf)])


def train_and_evaluate(train_df: pd.DataFrame, valid_df: pd.DataFrame) -> dict:
    X_train = train_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_train = train_df[TARGET]
    X_valid = valid_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_valid = valid_df[TARGET]

    pipe = build_pipeline()
    pipe.fit(X_train, y_train)
    model_preds = pipe.predict(X_valid)
    model_proba = pipe.predict_proba(X_valid)
    classes = pipe.named_steps["clf"].classes_

    baseline_preds = baseline_predict(valid_df)

    results = {
        "model_accuracy": accuracy_score(y_valid, model_preds),
        "model_macro_f1": f1_score(y_valid, model_preds, average="macro"),
        "baseline_accuracy": accuracy_score(y_valid, baseline_preds),
        "baseline_macro_f1": f1_score(y_valid, baseline_preds, average="macro"),
        "model_classification_report": classification_report(y_valid, model_preds, output_dict=True),
        "classes": list(classes),
    }
    return {
        "pipe": pipe,
        "results": results,
        "model_preds": model_preds,
        "model_proba": model_proba,
        "baseline_preds": baseline_preds,
    }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/home/claude/Den_Search_Ranking_Discoverability_Capstone")
    from src.features import build_train_valid

    raw = pd.read_csv(
        "/home/claude/Den_Search_Ranking_Discoverability_Capstone/data/sample_search_data.csv",
        parse_dates=["date"],
    )
    train_df, valid_df = build_train_valid(raw)
    out = train_and_evaluate(train_df, valid_df)
    r = out["results"]
    print(f"Baseline  -> accuracy: {r['baseline_accuracy']:.3f}  macro-F1: {r['baseline_macro_f1']:.3f}")
    print(f"Model     -> accuracy: {r['model_accuracy']:.3f}  macro-F1: {r['model_macro_f1']:.3f}")
