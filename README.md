# flyrank-capstone-den-search-ranking-discoverability

**Google Search Ranking & Discoverability Capstone — FlyRank AI Machine Learning Engineering Internship**  
**Lane chosen:** Refresh / Content Opportunity Scoring  
**Author:** Denmark Tagomata

---

Score pages that are growing, declining, recovering, or worth review, using a leakage-safe (out-of-time) validation design, and turn the result into a ranked, explainable action list with reason codes.

👉 **Deployed research paper:** see `submission/paper_url.txt` once deployed (instructions below), or open `paper/index.html` directly.

---

## Repository layout

```
Den_Search_Ranking_Discoverability_Capstone/
├── work/                                   # weekly assignment notebooks (the capstone build-up)
│   ├── 01_data_intake_and_eda.ipynb
│   ├── 02_duckdb_feature_aggregation.ipynb
│   ├── 03_baseline_and_model.ipynb
│   └── capstone_refresh_opportunity_scoring.ipynb   # the capstone notebook
├── src/                                    # shared library code used by every notebook
│   ├── data_gen.py         # public-safe synthetic data generator (stand-in for the gated warehouse)
│   ├── features.py         # DuckDB feature aggregation + leakage-safe label design
│   ├── scoring_model.py    # baseline + Random Forest model, trained & evaluated
│   ├── ranking_engine.py   # ranked action list + explainable reason codes
│   └── make_figures.py     # renders every chart used in the deployed paper
├── data/
│   └── sample_search_data.csv     # generated synthetic dataset (320 pages × 180 days)
├── outputs/
│   ├── figures/                   # PNG charts (also copied into paper/figures/)
│   └── ranked_recommendations.csv # full ranked action list, one row per page
├── paper/
│   ├── index.html                 # the deployed research paper (GitHub Pages entry point)
│   └── figures/                   # charts embedded in the paper
├── submission/
│   └── paper_url.txt              # <- put your deployed paper's URL here (mandatory)
├── requirements.txt
└── README.md
```

## Why the dataset is synthetic

The real FlyRank ML Internship warehouse is a **gated Hugging Face dataset**
requiring a personal read token (see the capstone brief's "Data" section).
This repository was built in a sandboxed environment with no network route to
Hugging Face, so `src/data_gen.py` generates a **public-safe synthetic
dataset** that reproduces the real warehouse's schema, time structure, and
realistic noise/seasonality. Every feature, the label definition, the
out-of-time validation split, the model, and the ranking engine are written
against that schema — so they run unchanged against the real data.

### Using the real FlyRank warehouse

Once you have your personal Hugging Face read token, the only change needed
is in `src/data_gen.py` / wherever you load the raw frame — replace the
synthetic loader with a DuckDB read straight from the warehouse, exactly as
taught in the program's Starter Notebook 03:

```python
import duckdb

raw = duckdb.sql("""
    SELECT page_id, date, impressions, clicks, avg_position,
           content_type, word_count, page_age_days
    FROM 'hf://datasets/<flyrank-warehouse-path>/*.parquet'
""").df()
```

Everything downstream (`src/features.py`, `src/scoring_model.py`,
`src/ranking_engine.py`, `src/make_figures.py`, and all four notebooks) reads
whichever DataFrame is passed to it and needs **no other changes**.

## Running it yourself

```bash
pip install -r requirements.txt

# 1. Generate the public-safe synthetic dataset
python src/data_gen.py

# 2. Reproduce the model, ranking engine, and every chart in the paper
python src/make_figures.py

# 3. (optional) re-run the notebooks end-to-end
jupyter nbconvert --to notebook --execute --inplace work/*.ipynb
```

## Deploying the paper to GitHub Pages

1. Push this repository to GitHub (see the step-by-step Git walkthrough your
   internship partner gave you alongside this deliverable).
2. In the repo on GitHub: **Settings → Pages → Build and deployment → Source:
   Deploy from a branch → Branch: `main`, folder: `/paper`** → Save.
3. GitHub gives you a URL like
   `https://<your-username>.github.io/Den_Search_Ranking_Discoverability_Capstone/`
   — wait 1–2 minutes for the first build.
4. Put that exact URL as the single line in `submission/paper_url.txt`, and
   update the "Full repository" link inside `paper/index.html`'s
   Reproducibility section to your real repo URL.
5. Commit and push the updated `submission/paper_url.txt` — this file is how
   the paper is found, per the capstone brief.

## Public-safety compliance

No client names, domains, URLs, private queries, credentials, raw exports, or
claims about a search engine's actual algorithm appear anywhere in this
repository, satisfying the capstone's public rule regardless of which dataset
(synthetic or real) backs the pipeline.
