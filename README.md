# ForeKart Inventory Intelligence

An interactive **Streamlit portfolio dashboard** for a **synthetic** e-commerce inventory analytics case study. This project does not use verified ForeKart company records.

## What the dashboard shows

- ARIMA versus Random Forest demand forecasts with matched-date MAE and RMSE
- Baseline, ARIMA, and Random Forest inventory policy simulations
- Stockout, excess-stock, rush-order and inventory-expense KPIs
- Website, Mobile App, and Marketplace performance
- Targeted promotion scenarios and sensitivity analysis
- CSV downloads and data quality notes

## Important accuracy notes

**10% fewer stockouts, 15% less overstock, 20% lower rush-order costs, and 25% lower inventory expenses are project targets, not automatically achieved results.**

The Module 7 simulation requires an audit/rerun to address mutually exclusive normal/rush ordering, outstanding orders at the simulation horizon, and cost accounting. Promotion response elasticities and fees in Module 8 are hypothetical. Do not present them as experimentally established savings.

## Run locally (Windows PowerShell)

```powershell
cd forekart_streamlit
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

If PowerShell blocks activation, skip activation and use:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open the localhost URL displayed in your terminal.

## Input files

Copy your actual outputs into `data/` using the exact names below. The dashboard gracefully shows a missing-file notice when an input is absent.

| File | Module |
|---|---|
| `forekart_cleaned.csv` | 3 |
| `forekart_all_arima_forecasts.csv` | 5 |
| `forekart_rf_forecasts.csv` | 6 |
| `forekart_inventory_simulation.csv` | 7 |
| `forekart_policy_kpis.csv` | 7 |
| `forekart_kpi_improvements.csv` | 7 |
| `forekart_channel_summary.csv` | 8 |
| `forekart_product_channel_kpis.csv` | 8 |
| `forekart_stock_coverage.csv` | 8 |
| `forekart_promotion_recommendations.csv` | 8 |
| `forekart_promotion_scenarios.csv` | 8 |

The app displays your exported data, not fabricated results. Some sections require more files than others.

## Deploy

1. Create a GitHub repository and push `app.py`, `requirements.txt`, `README.md`, and `data/` with **non-sensitive synthetic** CSVs.
2. Visit https://share.streamlit.io/ and sign in with GitHub.
3. Choose your repository, branch, and entrypoint `app.py`.
4. Deploy and test every dashboard section.

## Suggested repository layout

```
forekart_streamlit/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── data/
    └── (copy your module CSV outputs here)
```

## Resume / interview framing

"I built an end-to-end synthetic inventory analytics case study using Python, ARIMA, Random Forest, inventory policy simulation, and Streamlit. I compared forecast accuracy, evaluated inventory KPIs against a baseline, and analyzed channel-specific promotion scenarios. Business improvement targets were assessed using reproducible simulation outputs rather than assumed as achieved."
