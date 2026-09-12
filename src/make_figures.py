"""
make_figures.py
----------------
Renders the charts used in the deployed research paper's Results section.
Run after scoring_model.py / ranking_engine.py have been exercised once.
"""

import sys
sys.path.insert(0, "/home/claude/Den_Search_Ranking_Discoverability_Capstone")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay

from src.features import build_train_valid
from src.ranking_engine import build_ranked_recommendations
from src.scoring_model import train_and_evaluate

FIG_DIR = "/home/claude/Den_Search_Ranking_Discoverability_Capstone/outputs/figures"

GOLD = "#FFC908"
MAROON = "#8B1300"


def main():
    raw = pd.read_csv(
        "/home/claude/Den_Search_Ranking_Discoverability_Capstone/data/sample_search_data.csv",
        parse_dates=["date"],
    )
    train_df, valid_df = build_train_valid(raw)
    out = train_and_evaluate(train_df, valid_df)
    r = out["results"]

    # 1. Baseline vs model bar chart
    fig, ax = plt.subplots(figsize=(6, 4))
    metrics = ["Accuracy", "Macro F1"]
    baseline_vals = [r["baseline_accuracy"], r["baseline_macro_f1"]]
    model_vals = [r["model_accuracy"], r["model_macro_f1"]]
    x = range(len(metrics))
    width = 0.35
    ax.bar([i - width / 2 for i in x], baseline_vals, width, label="Rule-based baseline", color=MAROON)
    ax.bar([i + width / 2 for i in x], model_vals, width, label="Random Forest model", color=GOLD, edgecolor=MAROON)
    ax.set_xticks(list(x))
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.0)
    ax.set_title("Model vs. Baseline — Out-of-Time Validation Window")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/model_vs_baseline.png", dpi=150)
    plt.close(fig)

    # 2. Confusion matrix
    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay.from_predictions(
        valid_df["momentum_class"], out["model_preds"], ax=ax, cmap="Oranges", colorbar=False
    )
    ax.set_title("Model Confusion Matrix (Validation Window)")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/confusion_matrix.png", dpi=150)
    plt.close(fig)

    # 3. Ranked action distribution
    ranked = build_ranked_recommendations(
        valid_df, out["model_preds"], out["model_proba"], r["classes"]
    )
    counts = ranked["action"].value_counts().reindex(
        ["PRIORITY_REFRESH", "MONITOR", "PROTECT", "LOW_PRIORITY"]
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(counts.index, counts.values, color=[MAROON, GOLD, "#2E7D32", "#9E9E9E"])
    ax.set_title("Ranked Action Engine — Output Distribution")
    ax.set_ylabel("Number of pages")
    plt.xticks(rotation=15)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/action_distribution.png", dpi=150)
    plt.close(fig)

    # 4. Feature importance
    clf = out["pipe"].named_steps["clf"]
    pre = out["pipe"].named_steps["pre"]
    feat_names = list(pre.transformers_[0][2]) + list(
        pre.transformers_[1][1].get_feature_names_out(["content_type"])
    )
    importances = pd.Series(clf.feature_importances_, index=feat_names).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    importances.plot(kind="barh", ax=ax, color=GOLD, edgecolor=MAROON)
    ax.set_title("Model Feature Importance")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/feature_importance.png", dpi=150)
    plt.close(fig)

    ranked.to_csv(
        "/home/claude/Den_Search_Ranking_Discoverability_Capstone/outputs/ranked_recommendations.csv",
        index=False,
    )
    print("Figures written to", FIG_DIR)
    print(r)


if __name__ == "__main__":
    main()
