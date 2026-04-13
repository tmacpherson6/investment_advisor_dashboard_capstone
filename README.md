# Investment Portfolio Generator — MADS Capstone Project

## Overview

The Investment Portfolio Generator is an integrated decision framework that connects heterogeneous investor characteristics to a data-driven, institutionally-structured investment strategy. Given a brief investor questionnaire, the system estimates risk tolerance, constructs a constrained optimal portfolio from a universe of sector and fixed-income ETFs, and runs a Monte Carlo simulation to project the range of likely outcomes over the investor's chosen time horizon.

---

## Features

- **Risk Tolerance Scoring** — Estimates investor risk tolerance from a questionnaire using a mock heuristic formula (stub), producing a numeric score on a 1–10 scale and a volatility ceiling that constrains portfolio construction. Planned replacement: ordered logistic regression trained on SCF 2022 microdata.
- **LSTM Volatility Forecasting** — Annualised forward-looking volatility predictions (`ANN_SIG`) are sourced from a PyTorch LSTM model (Pete's series, 30-trial average). Historical vol from yfinance is retained separately for visualisation comparison.
- **Sharpe-Prior Expected Returns** — Expected returns (`ANN_MU`) are derived via a Sharpe-prior method: `μ_i = rf + sharpe_prior_i × σ_lstm_i`. This replaces naive historical mean returns, which were noisy and bull-market biased in the available window. Sharpe priors are anchored to Ilmanen (2011) and Erb & Harvey (2006).
- **Real Correlation Matrix** — Pearson correlation is estimated from full-history log returns downloaded live via yfinance (common start: December 2015, binding on XLRE launch). Positive semi-definiteness is enforced via eigenvalue clipping (Higham 2002). Parameters are cached for the calendar day.
- **Two-Tier Constrained Portfolio Optimisation** — Maximises the Sharpe ratio subject to:
  - *Tier 1 (asset class):* age/risk-score equity ceiling, vol band [floor, ceiling], inverse-risk-scaled GLD cap
  - *Tier 2 (intra-bucket):* sector concentration ≤30% of equity bucket, IG FI ≤50% of FI bucket, quadratic HYG scaling by risk score
- **Monte Carlo Simulation** — Simulates 1,000 portfolio paths over a user-selected horizon (1–30 years), reporting annualised return, volatility, Sharpe ratio, terminal value percentiles, and maximum drawdown.
- **Animated Visualisation** — Generates a spaghetti-plot GIF showing path convergence with live P5/P50/P95 overlays.
- **User Feedback Collection** — A `/api/feedback` endpoint stores model score feedback to PostgreSQL (if `DATABASE_URL` is configured) or falls back to a local JSONL file.
- **Technical Report** — A `/report` route serves a static methodology write-up page.

---

## Architecture

```
app.py          Flask application — routes, simulation, GIF generation
                Consumes all market parameters from logic.py via get_market_params()
                Owns mock_risk_score() stub (to be replaced by SCF model)

logic.py        Single source of truth for market parameters
                REAL:  Correlation matrix — full-history log returns (yfinance)
                REAL:  ANN_SIG — LSTM 30-trial average vol (annualized_volatility_predictions.csv)
                REAL:  ANN_SIG_HIST — historical vol from yfinance (retained for viz)
                REAL:  ANN_MU — Sharpe-prior method (rf + sharpe_prior_i * sigma_lstm_i)
                Exposes: get_market_params(), optimize_portfolio(), compute_equity_ceiling()
```

When the SCF risk model is ready, it plugs into `app.py`'s `mock_risk_score()`. `logic.py` requires no changes.

---

## ETF Universe

The portfolio is constructed from 17 instruments spanning full GICS sector coverage, the yield curve, credit, inflation protection, and gold.

| Ticker | Description | Asset Class |
|--------|-------------|-------------|
| XLF | Financials | Equity sector |
| XLK | Technology | Equity sector |
| XLU | Utilities | Equity sector |
| XLV | Health Care | Equity sector |
| XLE | Energy | Equity sector |
| XLI | Industrials | Equity sector |
| XLB | Materials | Equity sector |
| XLP | Consumer Staples | Equity sector |
| XLY | Consumer Discretionary | Equity sector |
| XLRE | Real Estate | Equity sector |
| BIL | 1–3 Month T-Bills | Fixed income — cash equivalent |
| IEF | 7–10yr Treasuries | Fixed income — intermediate |
| TLT | 20yr+ Treasuries | Fixed income — long duration |
| LQD | IG Corporate Credit | Fixed income — credit |
| HYG | High Yield Credit | Fixed income — risk-scaled |
| TIP | TIPS | Fixed income — inflation-linked |
| GLD | Gold | Real assets |

> **Note:** XLRE launched December 2015, setting the common start date for the correlation matrix. The 2008 GFC and 2011 European debt crisis are not represented in covariance estimates — a known limitation documented in the methodology.

---

## Portfolio Construction Methodology

### Equity Ceiling

The maximum equity allocation is determined by:

```
equity_ceiling = (100 - age) + (risk_score - 5) × 3
```

Capped at 90% for risk scores below 9 (fiduciary conservatism). Risk scores ≥9 allow the formula to run freely up to 100%. Floored at 10% to avoid degenerate all-cash portfolios.

### Intra-Bucket Constraints

| Constraint | Formula |
|------------|---------|
| Any single GICS sector | ≤ 30% of equity bucket |
| Any single IG FI instrument | ≤ 50% of FI bucket |
| HYG within FI bucket | ≤ (risk\_score / 10)² × 35% — quadratic scaling |
| GLD (absolute) | 15% − (risk\_score − 1) / 9 × 10% — inverse risk scaling |

### Vol Band

```
vol_floor = vol_ceiling × floor_pct
```

Where `floor_pct` is 25% for scores ≤4, 35% for scores ≤7, and 45% for scores >7. The floor encodes the fiduciary obligation to *deliver* the risk/return profile the client was assessed for, not merely cap downside.

### HYG Vol Floor

HYG is floored at 6% annualised volatility. The LSTM's limited feature set (ETF return + vol history only) misses credit spread dynamics that drive HYG risk. The floor is documented in the methodology.

### Optimiser

Scipy SLSQP with 5 random Dirichlet restarts to mitigate local minima on the non-convex Sharpe surface. Falls back to inverse-vol weighting if all restarts fail (logged as a warning).

---

## Model Status

| Component | Current state |
|-----------|--------------|
| Risk tolerance score | Mock heuristic formula in `app.py` — planned replacement: ordered logistic regression on SCF 2022 microdata |
| Volatility (ANN_SIG) | LSTM 30-trial average predictions (`annualized_volatility_predictions.csv`) |
| Expected returns (ANN_MU) | Sharpe-prior method: `rf + sharpe_prior_i × σ_lstm_i` |
| Correlation matrix | Full-history Pearson from yfinance daily close prices |

---

## Installation

### Prerequisites
- Python 3.11 or higher
- pip

### Setup

1. Clone the repository:
```bash
git clone https://github.com/tmacpherson6/investment_advisor_dashboard_capstone capstone
cd capstone
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate       # macOS/Linux
venv\Scripts\activate          # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

> **Note:** `requirements.txt` includes PyTorch (`torch==2.10.0`) which is required to run the LSTM volatility model notebooks. The Flask dashboard itself does not import PyTorch at runtime — it reads the pre-generated `annualized_volatility_predictions.csv`.

---

## Usage

### Running the Dashboard

1. Ensure `annualized_volatility_predictions.csv` is present in the repo root. This file is generated by running `Pete-D1_Annualized-Volatility.ipynb`. A copy is committed to the repository.

2. Activate your virtual environment and run:
```bash
python app.py
```

On startup, `logic.py` downloads historical price data from yfinance and computes the correlation matrix and historical volatility estimates. This takes approximately 10–20 seconds on first run; subsequent requests on the same day use the cached parameters.

3. Open `http://127.0.0.1:5000` in your browser.

> **Note:** The app runs with `debug=True` for local development. Do not expose it to the public internet in this state.

### Optional: Feedback Storage

Set the `DATABASE_URL` environment variable to a PostgreSQL connection string to persist user feedback to a database. Without it, feedback falls back to `feedback_log.jsonl` in the repo root.

---

## Project Structure

```
project/
├── app.py                          Flask app — routes, simulation, visualisation
├── logic.py                        Market parameters, LSTM vol loading, optimiser
├── data_prep.py                    Data preparation utilities for model training
├── model_testing_volatility.py     Volatility model evaluation and baseline comparison
├── models.py                       Model definitions
├── viz.py                          Visualisation utilities
│
├── templates/
│   ├── base.html                   Shared base layout
│   ├── index.html                  Investor questionnaire
│   ├── results.html                Portfolio output and simulation results
│   └── report.html                 Static technical report page
│
├── data/
│   ├── train-val-test/             Train/val/test splits (Pete's B-series pipeline)
│   └── downsample/W/train-val-test/ Downsampled weekly splits (Pete's C-series pipeline)
│
├── annualized_volatility_predictions.csv   LSTM vol output consumed by logic.py
├── yfinance_data.csv               Raw yfinance price data (generated by Tom's Notebook 3)
├── fred_data.csv                   FRED macro data (generated by Tom's Notebook 2)
├── raw_data_prediction_dataset.csv Combined prediction dataset (Tom's Notebook 4)
├── stationary_prediction_dataset.csv Stationarised dataset (Tom's Notebook 5)
├── SCFP2022.csv                    Survey of Consumer Finances 2022 microdata
├── volatility_results.csv          Volatility model evaluation results
├── volatility_baseline.json        Baseline metrics for vol model comparison
├── volatility_results.json         Full vol model results
│
├── requirements.txt
└── README.md
```

### Notebooks

All notebooks live in the repo root. They are organised by contributor and pipeline stage:

**Tom's notebooks — data pipeline and SCF exploration**
| Notebook | Purpose |
|----------|---------|
| Tom's Notebook 1 - Survey of Consumer Finances Exploration | EDA on SCF 2022 microdata |
| Tom's Notebook 2 - Connecting to St. Louis Federal Reserve | Downloads FRED macro data → `fred_data.csv` |
| Tom's Notebook 3 - yfinance web scrapping notebook | Downloads ETF price data → `yfinance_data.csv` |
| Tom's Notebook 4 - Creating Finalized Prediction Datasets | Merges FRED + yfinance → `raw_data_prediction_dataset.csv` |
| Tom's Notebook 5 - Transformed Finalized Prediction Datasets | Stationarity transforms → `stationary_prediction_dataset.csv` |

**Pete's notebooks — LSTM volatility model pipeline**
| Series | Notebooks | Purpose |
|--------|-----------|---------|
| A-series | Pete-A1 through Pete-A4 | Initial data inspection, hold-out split, early LSTM prototypes |
| B-series | Pete-B1 through Pete-B9 | Full feature engineering, train/val/test splits, Ridge/GBR/LSTM model development |
| C-series | Pete-C3 through Pete-C9 | Downsampled (weekly) pipeline — feature engineering, PCA, model variants |
| D-series | Pete-D1 | Annualised volatility aggregation → **`annualized_volatility_predictions.csv`** (consumed by `logic.py`) |

**Ryan's notebook**
| Notebook | Purpose |
|----------|---------|
| Ryan-SCF_Random_Forest_Risk_Pred | Random Forest risk tolerance prediction on SCF data |

**Kristine's notebooks**
| Notebook | Purpose |
|----------|---------|
| Kristine_SCF_Data_Prep_R | SCF data preparation (R kernel) |
| Kristine_SCF_Feature_Selection_R | Feature selection on SCF data (R kernel) |
| Kristine_SCF_Multiply_Imputed_Survey_Regression_R | Multiply imputed survey regression on SCF data (R kernel) |

---

## Technologies Used

- Python 3.11
- Flask
- yfinance
- NumPy / Pandas
- SciPy (SLSQP optimisation)
- Matplotlib / Pillow (GIF generation)
- PyTorch (LSTM volatility model)
- scikit-learn (Ridge, GBR, PCA, Random Forest)
- fredapi (FRED macro data)
- Jupyter / JupyterLab

---

## Known Limitations

- **Correlation history starts December 2015** due to XLRE launch date. The 2008 GFC and 2011 European debt crisis are not represented in covariance estimates.
- **Risk score is a stub.** The mock heuristic in `mock_risk_score()` produces a score from age, income, net worth, education, and experience using a simple linear formula. Allocations are illustrative until the SCF-trained model is integrated.
- **Normality assumption.** The Monte Carlo simulation uses multivariate normal shocks. Fat tails and skewness in actual sector returns are not captured.
- **Static correlation matrix.** Full-history Pearson correlation is used rather than time-varying estimation. Regime shifts in correlation structure are not reflected.
- **LSTM feature set.** The volatility model uses only ETF return and vol history, which may miss credit spread dynamics for HYG. A 6% vol floor compensates for this known blind spot.

---

## License

[License to be selected]

## Generative AI Disclosure
Claude (Anthropic, claude.ai) was used to assist in the generation of 
HTML, CSS, and portions of this README. All AI-generated content has 
been reviewed and validated by the authors. Inline disclosures are 
included in the relevant files where AI assistance was used.