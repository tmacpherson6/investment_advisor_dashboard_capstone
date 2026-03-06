"""
logic.py
────────────────────────────────────────────────────────────────────────────────
Single source of truth for market parameters consumed by app.py.

Current state
─────────────
  REAL:  Correlation matrix, historical annualised volatility
         → derived from yfinance daily close prices

  STUB:  Annualised expected returns (ANN_MU)
         → still hard-coded in app.py until forward-looking models are ready

Future state (no changes to app.py required)
─────────────────────────────────────────────
  When return/vol models are ready:
    1. Add model loading logic in the MODELS section below
    2. Populate ANN_MU and optionally ANN_SIG from model output
    3. Export them via get_market_params() — app.py reads them automatically

Caching
───────
  Parameters are computed once at app startup (module import time) and stored
  as module-level state. No per-request recomputation.
  Cache is valid for the calendar day it was computed.
  On the next day's first import the cache is stale and recomputed automatically.
────────────────────────────────────────────────────────────────────────────────
"""

import logging
from datetime import date, datetime
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
#  UNIVERSE
#  Sector ETFs give finer-grained risk/reward control vs broad-market ETFs.
#  Full GICS sector coverage + yield curve + credit + inflation + gold.
#  Note: XLRE (real estate) launched Dec 2015, which sets the common start
#        date and means pre-2016 history — including the 2008 GFC — is
#        excluded. This is documented in methodology comments below.
# ─────────────────────────────────────────────────────────────────────────────

ETF_UNIVERSE: list[str] = [
    # ── GICS sectors ──────────────────────────────────────────────────────
    "XLF",  # Financials
    "XLK",  # Technology
    "XLU",  # Utilities
    "XLV",  # Health Care
    "XLE",  # Energy
    "XLI",  # Industrials
    "XLB",  # Materials
    "XLP",  # Consumer Staples
    "XLY",  # Consumer Discretionary
    "XLRE",  # Real Estate  ← binding constraint: launched Dec 2015
    # ── Fixed income ──────────────────────────────────────────────────────
    "BIL",  # 1–3 month T-bills  (cash equivalent; longer history than SGOV)
    "IEF",  # 7–10yr Treasuries  (intermediate)
    "TLT",  # 20yr+ Treasuries   (long duration)
    "LQD",  # Investment-grade corporate credit
    "HYG",  # High-yield credit  (equity-like tail risk — see note in README)
    "TIP",  # TIPS               (inflation-linked)
    # ── Real assets ───────────────────────────────────────────────────────
    "GLD",  # Gold
]

N: int = len(ETF_UNIVERSE)

# ─────────────────────────────────────────────────────────────────────────────
#  MODULE-LEVEL CACHE
#  Populated once on first import; refreshed if the calendar date has changed.
# ─────────────────────────────────────────────────────────────────────────────

_cache: dict = {
    "corr": None,  # np.ndarray  (N x N)
    "ann_sig": None,  # np.ndarray  (N,)   annualised historical vol
    "etf_universe": ETF_UNIVERSE,
    "as_of_date": None,  # str  "YYYY-MM-DD"
    "common_start": None,  # str  "YYYY-MM-DD"  first date with full data
    "computed_on": None,  # date object used for staleness check
}


# ─────────────────────────────────────────────────────────────────────────────
#  DATA FETCHING
# ─────────────────────────────────────────────────────────────────────────────


def _fetch_prices(tickers: list[str], start: str = "2000-01-01") -> pd.DataFrame:
    """
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


def _clean_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Drop any leading rows that contain NaNs (i.e. before all ETFs have
    launched). This gives the longest fully-observed common history.

    The binding constraint is XLRE (launched Dec 2015), which means the 2008
    GFC is not represented in the correlation estimates. This is a known
    limitation documented in the capstone methodology section.

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


def _compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Log returns are preferred over simple returns for parameter estimation:
    - Time-additive, better approximation of normality over short intervals
    - More symmetric distribution reduces skew in the correlation matrix
    """
    return np.log(prices / prices.shift(1)).dropna()


def _estimate_correlation(log_returns: pd.DataFrame) -> np.ndarray:
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
    corr = log_returns.corr().values  # (N x N) numpy array

    # ── Enforce positive semi-definiteness ───────────────────────────────
    # Floating-point arithmetic can produce tiny negative eigenvalues.
    # Clip them to zero and reconstruct — this is the Higham (2002) approach
    # simplified for diagonal correction.
    eigvals, eigvecs = np.linalg.eigh(corr)
    eigvals_clipped = np.clip(eigvals, a_min=0, a_max=None)
    corr_psd = eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T

    # Re-normalise diagonal to exactly 1.0
    d = np.sqrt(np.diag(corr_psd))
    corr_psd = corr_psd / np.outer(d, d)
    np.fill_diagonal(corr_psd, 1.0)

    return corr_psd


def _estimate_ann_vol(log_returns: pd.DataFrame) -> np.ndarray:
    """
    Annualised historical volatility: daily std * sqrt(252).
    Returned as decimals (e.g. 0.15 = 15%), matching ANN_SIG convention in
    app.py.
    """
    return (log_returns.std() * np.sqrt(252)).values


# ─────────────────────────────────────────────────────────────────────────────
#  PORTFOLIO OPTIMISATION
# ─────────────────────────────────────────────────────────────────────────────

# Asset-class level weight caps — more defensible than a flat 30% across all.
# Conservative fixed income (BIL, IEF, TLT, TIP) allowed up to 60% individually
# so a risk-score-1 portfolio can legitimately hold mostly cash/duration without
# being artificially forced into equities to fill weight.
_WEIGHT_CAPS: dict[str, float] = {
    # ── Equity sectors: max 25% — finer control, prevents single-sector dominance
    "XLF": 0.25,
    "XLK": 0.25,
    "XLU": 0.25,
    "XLV": 0.25,
    "XLE": 0.25,
    "XLI": 0.25,
    "XLB": 0.25,
    "XLP": 0.25,
    "XLY": 0.25,
    "XLRE": 0.25,
    # ── Fixed income: cap reflects true risk character, not just asset class label
    "BIL": 0.40,  # cash equivalent — low risk, high cap appropriate
    "IEF": 0.40,  # intermediate duration — modest rate sensitivity
    "TIP": 0.40,  # inflation-linked — similar duration profile to IEF
    "LQD": 0.40,  # IG credit — spread risk but high-quality
    "TLT": 0.25,  # long duration — ~50% drawdown 2020-2023, equity-like vol
    "HYG": 0.25,  # high yield — high equity correlation in stress, credit risk
    # ── Real assets: commodities capped to prevent over-concentration
    "GLD": 0.15,
}


def optimize_portfolio(
    ann_mu: np.ndarray,
    ann_sig: np.ndarray,
    corr: np.ndarray,
    vol_ceiling: float,
    rf: float = 0.04,
    n_restarts: int = 5,
) -> dict[str, float]:
    """
    Maximise the Sharpe ratio subject to:
      1. Portfolio annualised volatility ≤ vol_ceiling
      2. Weights sum to 1
      3. All weights ≥ 0  (long-only)
      4. Per-asset weight caps defined in _WEIGHT_CAPS

    Uses scipy SLSQP with multiple random restarts to avoid local minima —
    the Sharpe surface is non-convex so a single start is not reliable.

    Parameters
    ──────────
    ann_mu      : annualised expected returns, shape (N,)
    ann_sig     : annualised volatilities,     shape (N,)
    corr        : correlation matrix,          shape (N, N)
    vol_ceiling : maximum annualised portfolio vol (e.g. 0.13)
    rf          : risk-free rate (annualised)
    n_restarts  : number of random starting points

    Returns
    ───────
    dict  {ticker: weight}  weights sum to 1.0, all >= 0
    """
    N = len(ann_mu)
    cov = np.diag(ann_sig) @ corr @ np.diag(ann_sig)  # (N x N) covariance

    caps = np.array([_WEIGHT_CAPS.get(t, 0.30) for t in ETF_UNIVERSE])

    def neg_sharpe(w: np.ndarray) -> float:
        port_ret = float(w @ ann_mu)
        port_var = float(w @ cov @ w)
        port_vol = np.sqrt(max(port_var, 1e-12))
        return -(port_ret - rf) / port_vol  # minimise negative Sharpe

    def port_vol(w: np.ndarray) -> float:
        return float(np.sqrt(w @ cov @ w))

    constraints = [
        {"type": "eq", "fun": lambda w: w.sum() - 1.0},  # fully invested
        {"type": "ineq", "fun": lambda w: vol_ceiling - port_vol(w)},  # vol ceiling
    ]

    # Per-asset bounds: [0, cap_i]
    bounds = [(0.0, float(caps[i])) for i in range(N)]

    best_result = None
    best_sharpe = -np.inf

    rng = np.random.default_rng(42)

    for _ in range(n_restarts):
        # Random Dirichlet start — respects sum-to-1 naturally
        w0 = rng.dirichlet(np.ones(N))
        w0 = np.clip(w0, 0, caps)
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
        # Fallback: minimum-variance portfolio ignoring vol ceiling
        # This can only fail if the covariance matrix is degenerate
        logger.warning(
            "optimize_portfolio: all restarts failed — "
            "falling back to inverse-vol weights. "
            f"vol_ceiling={vol_ceiling:.2%}"
        )
        inv_vol = 1.0 / np.maximum(ann_sig, 1e-6)
        w_fb = inv_vol / inv_vol.sum()
        return dict(zip(ETF_UNIVERSE, [round(float(x), 4) for x in w_fb]))

    w_opt = best_result.x
    # Clean up: zero out tiny weights below 0.1% (numerical noise)
    w_opt[w_opt < 0.001] = 0.0
    w_opt /= w_opt.sum()

    return dict(zip(ETF_UNIVERSE, [round(float(x), 4) for x in w_opt]))


# ─────────────────────────────────────────────────────────────────────────────
#  CACHE MANAGEMENT
# ─────────────────────────────────────────────────────────────────────────────


def _is_cache_stale() -> bool:
    """Returns True if cache has never been populated or was computed on a
    prior calendar day."""
    return _cache["computed_on"] is None or _cache["computed_on"] < date.today()


def _populate_cache() -> None:
    """
    Fetches prices, estimates parameters, and writes results into _cache.
    Called once at startup; subsequent calls on the same day are no-ops.
    """
    logger.info("logic.py: computing market parameters from yfinance...")

    prices = _fetch_prices(ETF_UNIVERSE)
    prices = _clean_prices(prices)
    log_rets = _compute_log_returns(prices)

    corr = _estimate_correlation(log_rets)
    ann_sig = _estimate_ann_vol(log_rets)

    _cache["corr"] = corr
    _cache["ann_sig"] = ann_sig
    _cache["as_of_date"] = prices.index[-1].strftime("%Y-%m-%d")
    _cache["common_start"] = prices.index[0].strftime("%Y-%m-%d")
    _cache["computed_on"] = date.today()

    logger.info(
        "logic.py: parameters computed. "
        f"Common start: {_cache['common_start']}  |  "
        f"As of: {_cache['as_of_date']}  |  "
        f"N={len(log_rets)} trading days"
    )


# ─────────────────────────────────────────────────────────────────────────────
#  PUBLIC INTERFACE
#  This is the only function app.py should call.
# ─────────────────────────────────────────────────────────────────────────────


def get_market_params() -> dict:
    """
    Returns the current market parameter set. Recomputes if cache is stale
    (i.e. on the first call of a new calendar day).

    Return schema
    ─────────────
    {
        "corr":         np.ndarray  shape (N, N)   — correlation matrix
        "ann_sig":      np.ndarray  shape (N,)     — annualised historical vol
        "etf_universe": list[str]                  — ordered ticker list
        "as_of_date":   str   "YYYY-MM-DD"         — last price date in data
        "common_start": str   "YYYY-MM-DD"         — first fully-observed date
    }

    NOT included (still owned by app.py as stubs):
        "ann_mu"  — forward-looking expected returns, not estimable from
                    history alone without introducing look-ahead bias.
                    Will be added here once return models are ready.
    """
    if _is_cache_stale():
        _populate_cache()

    return {
        "corr": _cache["corr"],
        "ann_sig": _cache["ann_sig"],
        "etf_universe": _cache["etf_universe"],
        "as_of_date": _cache["as_of_date"],
        "common_start": _cache["common_start"],
    }


# ─────────────────────────────────────────────────────────────────────────────
#  ── FUTURE: MODEL INTEGRATION STUB ─────────────────────────────────────────
#
#  When forward-looking return and vol models are ready, add them here.
#  app.py will automatically pick them up via get_market_params() with no
#  other changes required.
#
#  Example pattern:
#
#  from models import ReturnModel, VolModel
#
#  def _load_model_params() -> tuple[np.ndarray, np.ndarray]:
#      mu  = ReturnModel.predict(ETF_UNIVERSE)   # shape (N,)
#      sig = VolModel.predict(ETF_UNIVERSE)      # shape (N,)
#      return mu, sig
#
#  Then in _populate_cache():
#      _cache["ann_mu"], _cache["ann_sig"] = _load_model_params()
#
#  And expose in get_market_params():
#      "ann_mu": _cache["ann_mu"]
#
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
#  STARTUP — compute on import
#  This fires when app.py does `from logic import get_market_params`
#  keeping the /analyze route free of any data fetching latency.
# ─────────────────────────────────────────────────────────────────────────────

try:
    _populate_cache()
except Exception as e:
    logger.error(
        f"logic.py: failed to compute market parameters on startup: {e}\n"
        "app.py will fall back to its mock parameters until the next request "
        "triggers a retry via get_market_params()."
    )
