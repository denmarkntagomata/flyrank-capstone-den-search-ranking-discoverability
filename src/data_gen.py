"""
data_gen.py
-----------
Generates a PUBLIC-SAFE, synthetic daily search-performance dataset that
mirrors the *shape* of the FlyRank ML Internship warehouse (page-level daily
impressions, clicks, average position, and content metadata).

WHY SYNTHETIC DATA IS USED HERE
--------------------------------
The real FlyRank warehouse is distributed as a *gated* Hugging Face dataset
that requires a personal read token to access (see the capstone brief,
"Data" section). This sandboxed build environment has no network access to
huggingface.co, so the pipeline below was authored and validated against a
synthetic dataset that reproduces the same columns, time structure, and
realistic noise/seasonality patterns as the real warehouse.

Swapping in the real data requires no changes to the modeling logic — only
`load_dataframe()` in this file needs to be pointed at the real warehouse
(see README "Using the real FlyRank warehouse" section for the one-line
change using `duckdb.sql("... FROM 'hf://...'")`, exactly as taught in the
program's Starter Notebook 03).

No client names, domains, URLs, private queries, or credentials appear
anywhere in this repository, satisfying the capstone's public-safety rule
regardless of which dataset backs it.
"""

import numpy as np
import pandas as pd

RNG_SEED = 42
N_PAGES = 320
N_DAYS = 180
START_DATE = "2026-01-01"

CONTENT_TYPES = ["blog", "product", "landing", "guide", "category"]

# Rough "expected CTR by position" curve, loosely modeled on widely-published
# organic CTR studies. Used only to build a realistic, explainable feature
# (ctr_gap_vs_expected) — not attributed to any single proprietary source.
EXPECTED_CTR_BY_POSITION = {
    1: 0.28, 2: 0.15, 3: 0.10, 4: 0.07, 5: 0.05,
    6: 0.04, 7: 0.03, 8: 0.025, 9: 0.02, 10: 0.018,
}


def _expected_ctr(position: float) -> float:
    pos = int(round(min(max(position, 1), 20)))
    if pos <= 10:
        return EXPECTED_CTR_BY_POSITION[pos]
    return max(0.005, 0.018 - (pos - 10) * 0.0012)


def _simulate_page(rng: np.random.Generator, page_id: int, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Simulate one page's daily time series with one of four latent regimes:
    growing, declining, recovering, or stable — the ground-truth patterns the
    modeling stage is later asked to recover from noisy signals alone."""
    regime = rng.choice(["growing", "declining", "recovering", "stable"], p=[0.27, 0.27, 0.16, 0.30])
    content_type = rng.choice(CONTENT_TYPES)
    word_count = int(rng.normal(1400, 500))
    word_count = max(300, word_count)
    age_days = int(rng.uniform(60, 1200))
    base_position = rng.uniform(4, 18)
    base_impressions = rng.gamma(shape=2.0, scale=rng.uniform(40, 300))

    n = len(dates)
    t = np.arange(n)

    if regime == "growing":
        pos_trend = -base_position * 0.35 * (t / n)          # position improves (lower is better)
        imp_trend = 1.0 + 1.4 * (t / n)
    elif regime == "declining":
        pos_trend = base_position * 0.45 * (t / n)           # position worsens
        imp_trend = 1.0 - 0.55 * (t / n)
    elif regime == "recovering":
        # dips in the first half, recovers in the second half
        dip = -np.sin(np.pi * (t / n)) * base_position * 0.5
        pos_trend = -dip
        imp_trend = 1.0 + 0.5 * np.sin(np.pi * (t / n))
    else:  # stable
        pos_trend = np.zeros(n)
        imp_trend = np.ones(n)

    weekly_season = 1.0 + 0.08 * np.sin(2 * np.pi * t / 7.0)
    noise_pos = rng.normal(0, 0.6, size=n)
    noise_imp = rng.normal(1.0, 0.12, size=n)

    position = np.clip(base_position + pos_trend + noise_pos, 1.0, 40.0)
    impressions = np.clip(base_impressions * imp_trend * weekly_season * noise_imp, 0, None)
    impressions = np.round(impressions).astype(int)

    ctr_noise = rng.normal(1.0, 0.18, size=n)
    ctr = np.clip(np.array([_expected_ctr(p) for p in position]) * ctr_noise, 0.001, 0.9)
    clicks = np.round(impressions * ctr).astype(int)

    df = pd.DataFrame({
        "page_id": f"page_{page_id:04d}",
        "date": dates,
        "impressions": impressions,
        "clicks": clicks,
        "avg_position": np.round(position, 2),
        "content_type": content_type,
        "word_count": word_count,
        "page_age_days": age_days,
        "true_regime": regime,  # kept ONLY for offline evaluation notebooks; dropped before modeling
    })
    return df


def load_dataframe(n_pages: int = N_PAGES, n_days: int = N_DAYS, seed: int = RNG_SEED) -> pd.DataFrame:
    """Return the full public-safe synthetic warehouse as a long, tidy
    DataFrame: one row per (page_id, date)."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(START_DATE, periods=n_days, freq="D")
    frames = [_simulate_page(rng, i, dates) for i in range(n_pages)]
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    df = load_dataframe()
    out_path = "/home/claude/Den_Search_Ranking_Discoverability_Capstone/data/sample_search_data.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df):,} rows for {df.page_id.nunique()} pages to {out_path}")
