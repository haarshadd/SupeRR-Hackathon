# SupeRR Selector — Pressure Intelligence Dashboard

A compact Streamlit dashboard for the Round 1 pressure-performance analysis.

## Run locally

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL printed by Streamlit (normally http://localhost:8501).

## Included data

The `data/` directory contains derived ranking, validation, comparison, and coverage CSVs only. It does **not** contain the proprietary raw competition data pack or delivery-level source data.

## Dashboard sections

- Batter/bowler Top 10 shortlist and pressure metrics
- Searchable all-eligible-player explorer
- Pressure/ordinary blend weights
- Legacy model comparison
- Repeated split-half reliability results
- Win-probability model diagnostics and pressure coverage

## Deployment

This app can be deployed to Streamlit Community Cloud by pushing this folder to a GitHub repository and selecting `app.py` as the entry point. Keep the derived CSVs in `data/`. Do not add or upload the proprietary raw data pack, ball-by-ball data, or any source data restricted by the competition.

## Important interpretation

Ranks are model-based historical scouting signals, not forecasts or automatic auction recommendations. Repeated splits are not independent external validation. Player availability, fitness, squad balance, overseas slots, price and purse constraints require separate checks.
