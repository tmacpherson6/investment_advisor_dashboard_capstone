"""
logic.py - version 1.6
────────────────────────────────────────────────────────────────────────────────
Source for market parameters that will be used by our dashboard 'app.py'.

Current state of parameters
───────────────────────────
  REAL:  Correlation matrix is derived from yfinance daily close prices
  REAL:  Annualised volatility (ANN_SIG) is derived from LSTM predictions using 
         Pete's model (30-trial average, annualised: raw_pred * sqrt(252) * 100)
  REAL:  Expected returns (ANN_MU) this is where we have had to make the most
         significnat change as our LSTM models did not show improvement over dummy models. So we will use an expected return based on a Sharpe-prior method:
           mu_i = rf + sharpe_prior_i * sigma_lstm_i
         Rationale: LSTM vol forecasts beat VIX as baseline (lower MAE);
         return forecasting was not reliable, so we use economically-motivated
         Sharpe priors (literature and historical data) scaled by forward-looking vol rather than historical mean returns (which are noisy and bull-market biased in our window missing correction risks).

  NB:    HYG vol is floored at 6% annualised. Our model prediction is quite low 
         volatility for a 'Junk Bond' ETF. The LSTM's limited feature set
         (ETF return + vol history only) might be missing credit spread dynamics that drive HYG risk. Floor is documented in our methodology.

  NB:    ANN_SIG uses LSTM average vol for optimisation; historical vol from
         yfinance is retained separately for the vol comparison visualisation to show our predictions versus the historical volatility.

Caching (efficiency design)
────────────────────────────
  Parameters are computed once at app startup (module import time) and stored. No per-request recomputation saving valuable time.
  Cache is valid for the day it was first computed.
  On the next day's first import the cache is stale and will recompute automatically.
────────────────────────────────────────────────────────────────────────────────

Constraints for portfolio optimisation:

Real historical covariance structure from yfinance (correlation matrix)
LSTM forward-looking vol (30-trial avg) for ANN_SIG 
ANN_MU derivation from our Sharpe-prior method using the LSTM vol forecasts
Two-tier constraint system with fiduciary rationale 
Age-based equity ceiling with risk score adjustment
Vol band enforcing risk delivery obligation
HYG quadratic scaling within FI bucket (conservative risk allocation for high-yield credit in low-risk portfolios)
GLD inverse risk-score scaling (essentially a hedge)

"""

import logging
import os
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize

# Let's pull in the LSTM vol predictions from a CSV that Pete created
LSTM_VOL_CSV = Path(__file__).parent / "annualized_volatility_predictions.csv"

# Column names in the CSV
COL_AVG = "Average Volatility (test set)"
COL_LATEST = "Recent Volatility (latest prediction)"

# ─────────────────────────────────────────────────────────────────────────────
#  SHARPE PRIORS FOR RETURN DERIVATION
#  We need to calculate expected returns (ANN_MU) for the optimizer, but our 
#  LSTM return forecasts were not reliable. Instead, we use a Sharpe-prior 
#  method to derive ANN_MU from the LSTM vol forecasts and fixed Sharpe priors 
#  for each asset. 
#
#  SHARPE PRIORS  (used to derive ANN_MU = rf + sharpe * sigma_lstm)
#
#  Anchored to empirical long-run US equity Sharpe of ~0.4–0.5.
#  Fixed income and alternatives scaled relative to that anchor.
#  Sources: Ilmanen (2011), Erb & Harvey (2006), AQR research.
#  
#  CITATIONS:
#  Ilmanen, A. (2011). Expected returns: An investor’s guide to harvesting 
#  market rewards. Wiley.
#  Erb, C. B., & Harvey, C. R. (2006). The strategic and tactical value of 
#  commodity futures. Financial Analysts Journal, 62(2), 69–97. https://doi.org/
#  10.2469/faj.v62.n2.4084
# ─────────────────────────────────────────────────────────────────────────────

SHARPE_PRIORS = {
    "XLK":  0.50,  # Technology         — highest Sharpe sector historically
    "XLV":  0.45,  # Health Care        — defensive growth, resilient earnings
    "XLY":  0.42,  # Consumer Disc.     — cyclical but strong long-run returns
    "XLF":  0.40,  # Financials         — market-like; rate sensitivity nets out
    "XLI":  0.38,  # Industrials        — solid but cyclical
    "XLB":  0.32,  # Materials          — commodity-linked, higher vol/return
    "XLP":  0.30,  # Consumer Staples   — low vol but low return
    "XLRE": 0.30,  # Real Estate        — rate-sensitive; yield + limited capital gain
    "XLE":  0.28,  # Energy             — high vol, volatile commodity exposure
    "XLU":  0.25,  # Utilities          — bond-proxy; rate risk compresses Sharpe
    "LQD":  0.30,  # IG Credit          — yield pickup over Treasuries, modest vol
    "HYG":  0.28,  # High Yield         — equity-like tail risk erodes Sharpe
    "TIP":  0.25,  # TIPS               — real return protection, low nominal Sharpe
    "IEF":  0.20,  # Int. Treasuries    — duration risk for modest excess return
    "TLT":  0.12,  # Long Treasuries    — high duration risk
    "BIL":  0.10,  # T-Bills            — near-zero excess return by construction
    "GLD":  0.25,  # Gold               — diversifier; Erb & Harvey long-run estimate
}

# HYG vol floor: LSTM limited-feature model misses credit spread dynamics.
# 6% is conservative relative to HYG's typical 8–12% realised (historical) vol.
# this is where experience in the domain is crucial to identify and compensate for model blind spots which we should talk about in the report.
HYG_VOL_FLOOR_PCT = 6.0  

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
#  ETF UNIVERSE FOR PORTFOLIO OPTIMISATION
#  Sector ETFs give finer-grained risk/reward control vs broad-market ETFs like 
#  the SPY or QQQ.
#  We want to cover everything: sector coverage + yield curve + credit + 
#  inflation + gold.
#  Note: XLRE (real estate) launched Dec 2015, which sets the common start
#        date and means pre-2016 history — including the 2008 GFC — is
#        excluded. This is documented in methodology comments below.
# ─────────────────────────────────────────────────────────────────────────────

ETF_UNIVERSE = [
    # ── Sectors ──────────────────────────────────────────────────────
    "XLF",  # Financials
    "XLK",  # Technology
    "XLU",  # Utilities
    "XLV",  # Health Care
    "XLE",  # Energy
    "XLI",  # Industrials
    "XLB",  # Materials
    "XLP",  # Consumer Staples
    "XLY",  # Consumer Discretionary
    "XLRE",  # Real Estate with binding constraint: launched Dec 2015
    # ── Fixed income ──────────────────────────────────────────────────────
    "BIL",  # 1–3 month T-bills  (cash equivalent; longer history than SGOV)
    "IEF",  # 7–10yr Treasuries  (intermediate)
    "TLT",  # 20yr+ Treasuries   (long duration)
    "LQD",  # Investment-grade corporate credit
    "HYG",  # High-yield credit  (junk bonds with equity-like risk)
    "TIP",  # TIPS               (inflation-linked debt securities)
    # ── Real assets ───────────────────────────────────────────────────────
    "GLD",  # Gold               (Often a hedge/diversifier)
]

N = len(ETF_UNIVERSE)

# ─────────────────────────────────────────────────────────────────────────────
#  CACHE
#  Populated once on first import; refreshed if the calendar date has changed.
# ─────────────────────────────────────────────────────────────────────────────

cache = {
    "corr": None,          
    "ann_sig": None,       
    "ann_sig_hist": None,  
    "ann_mu": None,        
    "etf_universe": ETF_UNIVERSE,
    "as_of_date": None,    
    "common_start": None,  
    "computed_on": None,  
}


# ─────────────────────────────────────────────────────────────────────────────
#  DATA IMPORT
# ─────────────────────────────────────────────────────────────────────────────


def load_lstm_vol(tickers):
    """
    Load LSTM annualised volatility predictions from CSV.

    Returns (ann_sig_avg, ann_sig_latest) as decimal arrays aligned to `tickers`.
    Values in the CSV are percentages so they are divided by 100 here.

    HYG is floored at HYG_VOL_FLOOR_PCT to compensate for the limited-feature
    LSTM's inability to capture credit spread dynamics as discussed earlier.

    Raises FileNotFoundError if the CSV is missing, so the caller can fall back
    to historical vol gracefully which will hopefully never happen.
    """
    if not LSTM_VOL_CSV.exists():
        raise FileNotFoundError(
            f"LSTM vol CSV not found at {LSTM_VOL_CSV}. "
            "Falling back to historical volatility."
        )

    df = pd.read_csv(LSTM_VOL_CSV, index_col="ETF")

    sig_avg = []
    sig_latest = []
    for ticker in tickers:
        if ticker not in df.index:
            raise KeyError(
                f"Ticker {ticker} not found in LSTM vol CSV. "
                "Check that annualized_volatility_predictions.csv is up to date."
            )
        avg_pct    = float(df.loc[ticker, COL_AVG])
        latest_pct = float(df.loc[ticker, COL_LATEST])

        # Here is where we will apply HYG floor
        if ticker == "HYG":
            avg_pct    = max(avg_pct,    HYG_VOL_FLOOR_PCT)
            latest_pct = max(latest_pct, HYG_VOL_FLOOR_PCT)
            logger.info(f"HYG vol floored to {HYG_VOL_FLOOR_PCT}% (raw: avg={float(df.loc['HYG', COL_AVG]):.2f}%, latest={float(df.loc['HYG', COL_LATEST]):.2f}%)")

        # Convert from percentage to decimal
        sig_avg.append(avg_pct / 100.0)
        sig_latest.append(latest_pct / 100.0)

    return np.array(sig_avg), np.array(sig_latest)


def derive_ann_mu(ann_sig, tickers, rf):
    """
    Derive expected returns via Sharpe-prior method:
        mu_i = rf + sharpe_prior_i * sigma_lstm_i

    A minimum excess return floor of 0.1% above rf is applied to prevent
    any asset from being treated as strictly dominated by the risk-free rate,
    which would break the vol floor constraint for conservative portfolios. This is acceptable because it's assumed any rational investor would require at least some return premium to hold a risky asset, even one with very low Sharpe ratio, otherwise they might as well just hold cash.

    Args:
        ann_sig: annualised vol array (decimals), aligned to tickers
        tickers: ordered list matching ETF_UNIVERSE
        rf: risk-free rate (annualised, decimal)
    """
    mu = np.array([
        rf + SHARPE_PRIORS[t] * ann_sig[i]
        for i, t in enumerate(tickers)
    ])
    # Floor: no asset should be more than rf (would be zeroed out by optimizer)
    mu = np.maximum(mu, rf + 0.001)
    return mu

def fetch_prices(tickers, start="2005-01-01"):
    """
    This is an older function used to calculate historical return if we need it for comparison purposes, but we are not using it for the main ANN_MU derivation due to the reasons outlined in the docstring of the derive_ann_mu function.

    Download adjusted daily close prices for all tickers.

    Returns a DataFrame indexed by date with one column per ticker.
    Uses auto_adjust=True so splits/dividends are handled automatically.

    Raises RuntimeError if the download returns an empty frame (e.g. network
    outage) so the caller can fall back to cached/mock data gracefully.
    """
    raw = yf.download(
        tickers=tickers,
        start=start,
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    # yfinance returns MultiIndex columns when >1 ticker: (field, ticker)
    # Extract only the "Close" level
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]] if "Close" in raw.columns else raw

    if prices.empty:
        raise RuntimeError(
            "yfinance returned an empty DataFrame. "
            "Check network connectivity or ticker symbols."
        )

    # Enforce column order to match ETF_UNIVERSE
    prices = prices.reindex(columns=tickers)
    return prices


# ─────────────────────────────────────────────────────────────────────────────
#  CLEANING
# ─────────────────────────────────────────────────────────────────────────────


def clean_prices(prices):
    """
    Drop any leading rows that contain NaNs (i.e. before all ETFs have
    launched). This gives the longest fully-observed common history.

    The binding constraint is XLRE (launched Dec 2015), which means the 2008
    GFC is not represented in the correlation estimates. This is a known
    limitation.

    After dropping leading NaNs, forward-fill any remaining isolated gaps
    (e.g. staggered holidays) for at most 1 day, then drop any residual NaNs.
    """
    # Drop rows where ANY ticker is NaN (removes pre-launch period)
    prices = prices.dropna(how="any")

    if prices.empty:
        raise ValueError(
            "Price DataFrame is empty after dropping NaN rows. "
            "Check that all tickers in ETF_UNIVERSE have overlapping history."
        )

    return prices


# ─────────────────────────────────────────────────────────────────────────────
#  PARAMETER ESTIMATION
# ─────────────────────────────────────────────────────────────────────────────


def compute_log_returns(prices):
    """
    Log returns are preferred over simple returns for parameter estimation:
    - Time-additive, better approximation of normality over short intervals
    - More symmetric distribution reduces skew in the correlation matrix
    - Most commonly used in industry and academia for these purposes, so it aligns with standard practices and expectations in finance.
    """
    return np.log(prices / prices.shift(1)).dropna()


def estimate_correlation(log_returns):
    """
    Pearson correlation matrix from full-history log returns.

    Design choice: full history over EWMA.
    EWMA upweights recent observations, which biases correlations toward the
    most recent regime (often a stress period with artificially elevated
    equity-equity correlations). Full history averages across multiple regimes
    and produces a more conservative, diversification-preserving estimate.

    The resulting matrix is guaranteed to be symmetric. We enforce positive
    semi-definiteness via eigenvalue clipping as a safeguard.
    """
    corr = log_returns.corr().values  

    # ── Enforce positive semi-definiteness ───────────────────────────────
    # Floating-point arithmetic can produce tiny negative eigenvalues.
    # Clip them to zero and reconstruct — this is the Higham (2002) approach
    # simplified for diagonal correction.
    #
    # CITATION:
    # Higham, N. J. (2002). “Computing the nearest correlation matrix—A problem # from finance.” IMA Journal of Numerical Analysis, 22(3), 329-343.
    eigvals, eigvecs = np.linalg.eigh(corr)
    eigvals_clipped = np.clip(eigvals, a_min=0, a_max=None)
    corr_psd = eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T

    # Re-normalise diagonal to exactly 1.0
    d = np.sqrt(np.diag(corr_psd))
    corr_psd = corr_psd / np.outer(d, d)
    np.fill_diagonal(corr_psd, 1.0)

    return corr_psd


def estimate_ann_vol(log_returns):
    """
    Annualised historical volatility: daily std * sqrt(252).
    Returned as decimals (e.g. 0.15 = 15%), matching ANN_SIG convention in
    app.py. This will allow us to use the historical vol for the vol comparison visualisation without any scaling confusion, and it also matches the scale of the LSTM vol predictions which are also in decimals.
    """
    return (log_returns.std() * np.sqrt(252)).values


# ─────────────────────────────────────────────────────────────────────────────
#  PORTFOLIO OPTIMISATION — TWO-TIER CONSTRAINT SYSTEM
#
#  Tier 1 — Asset class buckets (absolute portfolio weights):
#    • Equity ceiling:     compute_equity_ceiling(age, risk_score)
#    • GLD absolute cap:   15%
#    • Vol band:           [vol_floor, vol_ceiling]
#
#  Tier 2 — Intra-bucket concentration (relative to bucket size):
#    • Any single sector ≤ 30% of the equity bucket to avoid piling into one
#    • Any single IG fixed income ETF ≤ 50% of the FI bucket
#    • HYG ≤ (risk_score/10)² × 35% of the FI bucket  (quadratic scaling)
#
#  Design rationale:
#    Tier-2 constraints prevent the optimizer exploiting a single instrument
#    within a bucket while respecting the bucket-level risk budget.
#    HYG scaling encodes the fiduciary view that high-yield credit is only
#    appropriate in proportion to demonstrated risk tolerance.
# ─────────────────────────────────────────────────────────────────────────────

# ── Asset-class classification ───────────────────────────────────────────────
EQUITY_TICKERS = {
    "XLF",
    "XLK",
    "XLU",
    "XLV",
    "XLE",
    "XLI",
    "XLB",
    "XLP",
    "XLY",
    "XLRE",
}

FI_TICKERS = {
    "BIL",
    "IEF",
    "TLT",
    "LQD",
    "HYG",
    "TIP",
}

# Investment-grade fixed income subset for intra-bucket constraints (no HYG)
IG_FI_TICKERS = {
    "BIL",
    "IEF",
    "TLT",
    "LQD",
    "TIP",
}

# Index masks aligned to ETF_UNIVERSE order
EQUITY_MASK = np.array(
    [1.0 if t in EQUITY_TICKERS else 0.0 for t in ETF_UNIVERSE]
)

# Not sure we will use this one but let's make it incase
FI_MASK = np.array(
    [1.0 if t in FI_TICKERS else 0.0 for t in ETF_UNIVERSE]
)

# Per-instrument index lookups for intra-bucket constraints
EQUITY_INDICES = [
    i for i, t in enumerate(ETF_UNIVERSE) if t in EQUITY_TICKERS
]
IG_FI_INDICES = [
    i for i, t in enumerate(ETF_UNIVERSE) if t in IG_FI_TICKERS
]
HYG_IDX = ETF_UNIVERSE.index("HYG")
GLD_IDX = ETF_UNIVERSE.index("GLD")

# Set our absolute per-instrument upper bounds (used as scipy bounds,
# not constraints)
# These are loose safety rails — the intra-bucket constraints should do the 
# real work
ABS_CAPS = {
    "XLF": 0.40,
    "XLK": 0.40,
    "XLU": 0.40,
    "XLV": 0.40,
    "XLE": 0.40,
    "XLI": 0.40,
    "XLB": 0.40,
    "XLP": 0.40,
    "XLY": 0.40,
    "XLRE": 0.40,
    "BIL": 0.60,
    "IEF": 0.60,
    "TLT": 0.50,
    "LQD": 0.60,
    "HYG": 0.40,
    "TIP": 0.60,
    "GLD": 0.15,  
}

# Keep WEIGHT_CAPS as public alias for any external references just in case
WEIGHT_CAPS = ABS_CAPS


def compute_equity_ceiling(age, risk_score):
    """
    Tier-1 equity ceiling as a fraction of total portfolio.
    
    The typical formula in industry for a starting point is 100-age.

    Our Formula:
    Formula: (100 - age) + (risk_score - 5) * 3

    Design rationale
    ────────────────
    - 100-age baseline: classic rule of thumb for equity allocation, reflecting decreasing risk capacity with age.
    - Risk score adjustment: ±15% swing around the age baseline.
    - Hard ceiling: 90% for risk_score < 9 (fiduciary conservatism).
    - Risk score >= 9: formula runs freely up to 100%.
    - Floor: 10% minimum equity regardless of age/score.
    """
    raw = (100 - age) + (risk_score - 5) * 3
    if risk_score >= 9:
        ceiling_pct = float(np.clip(raw, 10, 100))
    else:
        ceiling_pct = float(np.clip(raw, 10, 90))
    return ceiling_pct / 100.0


def build_constraints(
    eq_ceiling,
    vol_floor,
    vol_ceiling,
    fi_bucket,
    risk_score,
    cov,
    gld_cap,
):
    """
    Builds the full two-tier constraint list for us to use with scipy.optimize.minimize.

    Tier-1 constraints (absolute portfolio weights):
      1. Weights must sum to 1
      2. vol_floor ≤ portfolio vol ≤ vol_ceiling
      3. Equity bucket ≤ eq_ceiling
      4. GLD ≤ gld_cap  (inverse risk-score scaled: 15% → 5%)

    Tier-2 constraints (intra-bucket concentration):
      5. Each sector ≤ 30% of equity bucket
      6. Each IG FI instrument ≤ 50% of FI bucket
      7. HYG ≤ (risk_score/10)² × 35% of FI bucket  (quadratic)
    """
    # ── Tier-2 derived limits ─────────────────────────────────────────────
    sector_cap = eq_ceiling * 0.30  # 30% of equity bucket
    ig_fi_cap = fi_bucket * 0.50  # 50% of FI bucket
    hyg_cap = fi_bucket * (risk_score / 10) ** 2 * 0.35  # quadratic

    def port_vol(w):
        return float(np.sqrt(w @ cov @ w))

    def equity_sum(w):
        return float(w @ EQUITY_MASK)

    constraints = [
        # ── Tier 1 ───────────────────────────────────────────────────────
        {"type": "eq", "fun": lambda w: w.sum() - 1.0},
        {"type": "ineq", "fun": lambda w: vol_ceiling - port_vol(w)},
        {"type": "ineq", "fun": lambda w: port_vol(w) - vol_floor},
        {"type": "ineq", "fun": lambda w: eq_ceiling - equity_sum(w)},
        {"type": "ineq", "fun": lambda w: gld_cap - w[GLD_IDX]},  # GLD inverse-scaled
        # ── Tier 2: intra-equity sector caps ─────────────────────────────
        *[
            {"type": "ineq", "fun": lambda w, i=i: sector_cap - w[i]}
            for i in EQUITY_INDICES
        ],
        # ── Tier 2: intra-FI IG caps ─────────────────────────────────────
        *[
            {"type": "ineq", "fun": lambda w, i=i: ig_fi_cap - w[i]}
            for i in IG_FI_INDICES
        ],
        # ── Tier 2: HYG quadratic risk-scaled cap ────────────────────────
        {"type": "ineq", "fun": lambda w: hyg_cap - w[HYG_IDX]},
    ]
    return constraints


def optimize_portfolio(
    ann_mu,
    ann_sig,
    corr,
    vol_ceiling,
    age,
    risk_score,
    rf = 0.04, # default to around 4%
    n_restarts = 5,
):
    """
    Maximise Sharpe ratio subject to the two-tier constraint system as described above.

    Tier 1 — Asset class:
      vol_floor ≤ portfolio vol ≤ vol_ceiling
      Equity ≤ compute_equity_ceiling(age, risk_score)
      GLD ≤ 15% - (risk_score-1)/9 * 10%  (inverse-scaled: score 1→15%, score 10→5%)

    Tier 2 — Intra-bucket:
      Any GICS sector ≤ 30% of equity bucket
      Any IG FI instrument ≤ 50% of FI bucket
      HYG ≤ (risk_score/10)² × 35% of FI bucket

    Vol floor scaling (fraction of vol_ceiling):
      risk_score ≤ 4 → 25%  |  ≤ 7 → 35%  |  > 7 → 45%
    """
    N = len(ann_mu)
    cov = np.diag(ann_sig) @ corr @ np.diag(ann_sig)
    abs_caps = np.array([ABS_CAPS.get(t, 0.40) for t in ETF_UNIVERSE])
    eq_ceiling = compute_equity_ceiling(age, risk_score)

    # ── GLD cap: inverse risk-score scaling ──────────────────────────────
    # Conservative clients benefit most from gold as safe haven/inflation hedge.
    # Aggressive clients should deploy that capital into growth equity instead.
    # Score 1 → 15%  |  Score 5 → 10.6%  |  Score 10 → 5%
    gld_cap = 0.15 - (risk_score - 1) / 9 * 0.10

    fi_bucket = max(1.0 - eq_ceiling - gld_cap, 0.05)

    # ── Vol floor ─────────────────────────────────────────────────────────
    if risk_score <= 4:
        floor_pct = 0.25
    elif risk_score <= 7:
        floor_pct = 0.35
    else:
        floor_pct = 0.45
    vol_floor = vol_ceiling * floor_pct

    logger.info(
        f"optimize_portfolio: age={age}, risk_score={risk_score} | "
        f"eq_ceiling={eq_ceiling:.1%}, fi_bucket≈{fi_bucket:.1%}, gld_cap={gld_cap:.1%} | "
        f"vol=[{vol_floor:.1%}, {vol_ceiling:.1%}] | "
        f"HYG_cap={fi_bucket * (risk_score / 10) ** 2 * 0.35:.1%}"
    )

    constraints = build_constraints(
        eq_ceiling, vol_floor, vol_ceiling, fi_bucket, risk_score, cov, gld_cap
    )
    bounds = [(0.0, float(abs_caps[i])) for i in range(N)]

    def neg_sharpe(w: np.ndarray) -> float:
        port_ret = float(w @ ann_mu)
        port_var = float(w @ cov @ w)
        port_vol = np.sqrt(max(port_var, 1e-12))
        return -(port_ret - rf) / port_vol

    best_result = None
    best_sharpe = -np.inf
    rng = np.random.default_rng(42)

    for _ in range(n_restarts):
        w0 = rng.dirichlet(np.ones(N))
        w0 = np.clip(w0, 0, abs_caps)
        w0 /= w0.sum()

        res = minimize(
            neg_sharpe,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-9, "maxiter": 1000},
        )

        if res.success and -res.fun > best_sharpe:
            best_sharpe = -res.fun
            best_result = res

    if best_result is None or not best_result.success:
        logger.warning(
            f"optimize_portfolio: all restarts failed — falling back to "
            f"inverse-vol weights. vol=[{vol_floor:.2%},{vol_ceiling:.2%}], "
            f"eq_ceiling={eq_ceiling:.2%}, age={age}, risk_score={risk_score}"
        )
        inv_vol = 1.0 / np.maximum(ann_sig, 1e-6)
        w_fb = inv_vol / inv_vol.sum()
        return dict(zip(ETF_UNIVERSE, [round(float(x), 4) for x in w_fb]))

    w_opt = best_result.x
    w_opt[w_opt < 0.001] = 0.0
    w_opt /= w_opt.sum()

    return dict(zip(ETF_UNIVERSE, [round(float(x), 4) for x in w_opt]))


# ─────────────────────────────────────────────────────────────────────────────
#  CACHE MANAGEMENT
# ─────────────────────────────────────────────────────────────────────────────


def is_cache_stale():
    """Returns True if cache has never been populated or was computed on a
    prior calendar day."""
    return cache["computed_on"] is None or cache["computed_on"] < date.today()


def populate_cache():
    """
    Fetches prices, loads LSTM vol, estimates parameters, writes into cache.
    Called once at startup; subsequent calls on the same day don't need to re-run.

    Vol priority:
      ANN_SIG (primary) → LSTM 30-trial average vol (forward-looking)
      ANN_SIG_HIST      → yfinance historical vol (retained for viz comparison)
      Fallback           → historical vol used for both if LSTM CSV unavailable
    """
    logger.info("logic.py: computing market parameters...")

    prices = fetch_prices(ETF_UNIVERSE)
    prices = clean_prices(prices)
    log_rets = compute_log_returns(prices)

    corr = estimate_correlation(log_rets)
    ann_sig_hist = estimate_ann_vol(log_rets)

    # ── Load LSTM vol (primary) ───────────────────────────────────────────
    rf = 0.04  # Must match RF_ANN in app.py usually around 4% for consistency in ANN_MU derivation
    try:
        ann_sig_lstm_avg, _ = load_lstm_vol(ETF_UNIVERSE)
        ann_sig = ann_sig_lstm_avg
        logger.info("logic.py: using LSTM volatility predictions (30-trial avg).")
    except (FileNotFoundError, KeyError) as e:
        logger.warning(f"logic.py: LSTM vol unavailable ({e}). Falling back to historical vol.")
        ann_sig = ann_sig_hist

    ann_mu = derive_ann_mu(ann_sig, ETF_UNIVERSE, rf)

    cache["corr"]          = corr
    cache["ann_sig"]       = ann_sig
    cache["ann_sig_hist"]  = ann_sig_hist
    cache["ann_mu"]        = ann_mu
    cache["as_of_date"]    = prices.index[-1].strftime("%Y-%m-%d")
    cache["common_start"]  = prices.index[0].strftime("%Y-%m-%d")
    cache["computed_on"]   = date.today()

    logger.info(
        "logic.py: parameters ready. "
        f"Common start: {cache['common_start']}  |  "
        f"As of: {cache['as_of_date']}  |  "
        f"N={len(log_rets)} trading days"
    )
    # Log implied return summary for an audit trail
    for i, t in enumerate(ETF_UNIVERSE):
        logger.info(
            f"  {t}: vol={ann_sig[i]*100:.2f}% (hist={ann_sig_hist[i]*100:.2f}%)  "
            f"mu={ann_mu[i]*100:.2f}%  sharpe_prior={SHARPE_PRIORS[t]}"
        )


# ─────────────────────────────────────────────────────────────────────────────
#  APP.PY INTEGRATION
#  Let's create a single function that app.py should call to get all this info
# ─────────────────────────────────────────────────────────────────────────────


def get_market_params():
    """
    Returns the current market parameter set. Recomputes if cache is stale.

    Returns
    ───────
    {
        "corr":         np.ndarray  shape (N, N)   — correlation matrix (historical)
        "ann_sig":      np.ndarray  shape (N,)     — LSTM annualised vol (primary, decimals)
        "ann_sig_hist": np.ndarray  shape (N,)     — historical vol from yfinance (for viz)
        "ann_mu":       np.ndarray  shape (N,)     — implied returns via Sharpe priors (decimals)
        "etf_universe": list[str]                  — ordered ticker list
        "as_of_date":   str   "YYYY-MM-DD"         — last price date in yfinance data
        "common_start": str   "YYYY-MM-DD"         — first fully-observed date
    }
    """
    if is_cache_stale():
        populate_cache()

    return {
        "corr":         cache["corr"],
        "ann_sig":      cache["ann_sig"],
        "ann_sig_hist": cache["ann_sig_hist"],
        "ann_mu":       cache["ann_mu"],
        "etf_universe": cache["etf_universe"],
        "as_of_date":   cache["as_of_date"],
        "common_start": cache["common_start"],
    }

# ─────────────────────────────────────────────────────────────────────────────
#  STARTUP — compute on import
#  This fires when app.py does `from logic import get_market_params`
#  keeping the /analyze route free of any data fetching latency.
# ─────────────────────────────────────────────────────────────────────────────

try:
    populate_cache()
except Exception as e:
    logger.error(
        f"logic.py: failed to compute market parameters on startup: {e}\n"
        "app.py will fall back to its mock parameters until the next request "
        "triggers a retry via get_market_params()."
    )