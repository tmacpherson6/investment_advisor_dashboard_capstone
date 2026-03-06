# Investment Portfolio Generator — MADS Capstone Project

## Overview

The Investment Portfolio Generator is an integrated decision framework that connects heterogeneous investor characteristics to a data-driven, institutionally-structured investment strategy. Given a brief investor questionnaire, the system estimates risk tolerance, constructs a constrained optimal portfolio from a universe of sector and fixed income ETFs, and runs a Monte Carlo simulation to project the range of likely outcomes over the investor's chosen time horizon.

The system is designed with a clean separation between real market data, stub parameters, and future model outputs — making it straightforward to swap in production-grade return and volatility models as they are developed.

---

## Features

- **Risk Tolerance Scoring** — Estimates investor risk tolerance from a questionnaire using a mock ordered logistic regression model (stub), producing a numeric score and a volatility ceiling that constrains portfolio construction.
- **Real Market Parameter Estimation** — Pulls daily adjusted close prices from yfinance at startup, computes a full-history log-return correlation matrix and annualised historical volatilities for all 17 instruments. Parameters are cached for the calendar day and refreshed automatically on next-day startup.
- **Two-Tier Constrained Portfolio Optimisation** — Maximises the Sharpe ratio subject to a two-tier constraint system:
  - *Tier 1 (asset class):* age/risk-score-based equity ceiling, vol band [floor, ceiling], and inverse-risk-scaled GLD cap
  - *Tier 2 (intra-bucket):* sector concentration limits (≤30% of equity bucket), IG fixed income limits (≤50% of FI bucket), and quadratic HYG scaling by risk score
- **Monte Carlo Simulation** — Simulates 1,000 portfolio paths over a user-selected horizon (1–30 years), reporting annualised return, volatility, Sharpe ratio, terminal value percentiles, and maximum drawdown.
- **Animated Visualisation** — Generates a spaghetti-plot GIF showing path convergence with live P5/P50/P95 overlays.

---

## Architecture

```
app.py          Flask application — routes, simulation, GIF generation
                Owns stub ANN_MU (forward-looking expected returns)
                Consumes all market parameters from logic.py

logic.py        Single source of truth for market parameters
                REAL:  correlation matrix, annualised historical vol (yfinance)
                STUB:  ANN_MU will be replaced by GARCH/LSTM models
                Exposes: get_market_params(), optimize_portfolio(), compute_equity_ceiling()
```

When return and volatility models are ready, only `logic.py` changes — `app.py` requires no modification.

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

> **Note:** XLRE launched December 2015, setting the common start date for the correlation matrix. The 2008 GFC is therefore not represented in historical covariance estimates — acknowledged as a known limitation in the methodology.

---

## Portfolio Construction Methodology

### Equity Ceiling

The maximum equity allocation is determined by:

```
equity_ceiling = (110 - age) + (risk_score - 5) × 3
```

Capped at 90% for risk scores below 9 (fiduciary conservatism). Risk scores ≥ 9 allow the formula to run freely up to 100%. Floored at 10% to avoid degenerate all-cash portfolios.

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

### Optimiser

Scipy SLSQP with 5 random Dirichlet restarts to mitigate local minima on the non-convex Sharpe surface. Falls back to inverse-vol weighting if all restarts fail (logged as a warning).

---

## Planned Model Integration

| Component | Current state | Planned replacement |
|-----------|--------------|-------------------|
| Risk tolerance score | Mock heuristic formula | Ordered logistic regression on SCF 2022 microdata |
| Expected returns (ANN_MU) | Stub values in `app.py` | LSTM model — sector return forecasting |
| Volatility (ANN_SIG) | Historical from yfinance | GARCH model — conditional vol forecasting |

When models are ready, they are imported in `logic.py` and exposed via `get_market_params()`. `app.py` requires no changes.

---

## Installation

### Prerequisites
- Python 3.11 or higher
- pip

### Setup

1. Clone the repository:
```bash
git clone https://github.com/tmacpherson6/investment_advisor_dashboard_capstone capstone
```

2. Create and activate a virtual environment:
```bash
python -m venv capstone
source capstone/bin/activate       # macOS/Linux
capstone\Scripts\activate          # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

---

## Usage

1. Activate your virtual environment (see Setup above).

2. Navigate to the project folder:
```bash
cd path/to/capstone
```

3. Run the Flask app:
```bash
python app.py
```

On startup, `logic.py` will download historical price data from yfinance and compute the correlation matrix and volatility estimates. This takes approximately 10–20 seconds on first run; subsequent requests on the same day use the cached parameters.

4. Open `http://127.0.0.1:5000` in your browser.

> **Note:** The app runs with `debug=True` for local development. Do not expose it to the public internet in this state.

---

## Project Structure

```
project/
├── app.py                  Flask app — routes, simulation, visualisation
├── logic.py                Market parameters, optimiser, constraint system
├── templates/
│   ├── index.html          Investor questionnaire
│   └── results.html        Portfolio output and simulation results
├── notebooks/              Exploratory analysis and model development
├── documentation/          Methodology write-ups
├── requirements.txt
└── README.md
```

---

## Technologies Used

- Python 3.11
- Flask
- yfinance
- NumPy / Pandas
- SciPy (SLSQP optimisation)
- Matplotlib / Pillow (GIF generation)
- PyTorch (LSTM — planned)
- arch (GARCH — planned)
- FRED API (macro features — planned)

---

## Known Limitations

- **Correlation history starts December 2015** due to XLRE launch date. The 2008 GFC and 2011 European debt crisis are not represented in covariance estimates.
- **Expected returns are stubs.** Portfolio weights are sensitive to return assumptions; current allocations are illustrative until models are integrated.
- **Normality assumption.** The Monte Carlo simulation uses multivariate normal shocks. Fat tails and skewness in actual sector returns are not captured.
- **Static correlation matrix.** Full-history Pearson correlation is used rather than time-varying (DCC-GARCH) estimation. Regime shifts in correlation structure are not reflected.

---

## License

[License to be selected]