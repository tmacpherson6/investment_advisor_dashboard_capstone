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

How we have constrained so far:

Real historical covariance and vol from yfinance
Two-tier constraint system with fiduciary rationale
Age-based equity ceiling with risk score adjustment
Vol band enforcing risk delivery obligation
HYG quadratic scaling within FI bucket
GLD inverse risk-score scaling

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
#  PORTFOLIO OPTIMISATION — TWO-TIER CONSTRAINT SYSTEM
#
#  Tier 1 — Asset class buckets (absolute portfolio weights):
#    • Equity ceiling:     compute_equity_ceiling(age, risk_score)
#    • GLD absolute cap:   15%
#    • Vol band:           [vol_floor, vol_ceiling]
#
#  Tier 2 — Intra-bucket concentration (relative to bucket size):
#    • Any single GICS sector ≤ 30% of the equity bucket
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
_EQUITY_TICKERS: set[str] = {
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

_FI_TICKERS: set[str] = {
    "BIL",
    "IEF",
    "TLT",
    "LQD",
    "HYG",
    "TIP",
}

_IG_FI_TICKERS: set[str] = {  # investment-grade only — HYG excluded
    "BIL",
    "IEF",
    "TLT",
    "LQD",
    "TIP",
}

# Index masks aligned to ETF_UNIVERSE order
_EQUITY_MASK: np.ndarray = np.array(
    [1.0 if t in _EQUITY_TICKERS else 0.0 for t in ETF_UNIVERSE]
)
_FI_MASK: np.ndarray = np.array(
    [1.0 if t in _FI_TICKERS else 0.0 for t in ETF_UNIVERSE]
)

# Per-instrument index lookups for intra-bucket constraints
_EQUITY_INDICES: list[int] = [
    i for i, t in enumerate(ETF_UNIVERSE) if t in _EQUITY_TICKERS
]
_IG_FI_INDICES: list[int] = [
    i for i, t in enumerate(ETF_UNIVERSE) if t in _IG_FI_TICKERS
]
_HYG_IDX: int = ETF_UNIVERSE.index("HYG")
_GLD_IDX: int = ETF_UNIVERSE.index("GLD")

# Absolute per-instrument upper bounds (used as scipy bounds, not constraints)
# These are loose safety rails — the intra-bucket constraints do the real work
_ABS_CAPS: dict[str, float] = {
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
    "GLD": 0.15,  # loose bound — dynamic cap enforced via constraint in _build_constraints
}

# Keep _WEIGHT_CAPS as public alias for any external references
_WEIGHT_CAPS = _ABS_CAPS


def compute_equity_ceiling(age: int, risk_score: float) -> float:
    """
    Tier-1 equity ceiling as a fraction of total portfolio.

    Formula: (110 - age) + (risk_score - 5) * 3

    Design rationale
    ────────────────
    - 110-age baseline: updated life-expectancy rule vs classic 100-age.
    - Risk score adjustment: ±15% swing around the age baseline.
    - Hard ceiling: 90% for risk_score < 9 (fiduciary conservatism).
    - Risk score >= 9: formula runs freely up to 100%.
    - Floor: 10% minimum equity regardless of age/score.
    """
    raw = (110 - age) + (risk_score - 5) * 3
    if risk_score >= 9:
        ceiling_pct = float(np.clip(raw, 10, 100))
    else:
        ceiling_pct = float(np.clip(raw, 10, 90))
    return ceiling_pct / 100.0


def _build_constraints(
    eq_ceiling: float,
    vol_floor: float,
    vol_ceiling: float,
    fi_bucket: float,
    risk_score: float,
    cov: np.ndarray,
    gld_cap: float,
) -> list[dict]:
    """
    Builds the full two-tier constraint list for scipy.optimize.minimize.

    Tier-1 constraints (absolute portfolio weights):
      1. Weights sum to 1
      2. vol_floor ≤ portfolio vol ≤ vol_ceiling
      3. Equity bucket ≤ eq_ceiling
      4. GLD ≤ gld_cap  (inverse risk-score scaled: 15% → 5%)

    Tier-2 constraints (intra-bucket concentration):
      5. Each GICS sector ≤ 30% of equity bucket
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
        return float(w @ _EQUITY_MASK)

    constraints = [
        # ── Tier 1 ───────────────────────────────────────────────────────
        {"type": "eq", "fun": lambda w: w.sum() - 1.0},
        {"type": "ineq", "fun": lambda w: vol_ceiling - port_vol(w)},
        {"type": "ineq", "fun": lambda w: port_vol(w) - vol_floor},
        {"type": "ineq", "fun": lambda w: eq_ceiling - equity_sum(w)},
        {"type": "ineq", "fun": lambda w: gld_cap - w[_GLD_IDX]},  # GLD inverse-scaled
        # ── Tier 2: intra-equity sector caps ─────────────────────────────
        *[
            {"type": "ineq", "fun": lambda w, i=i: sector_cap - w[i]}
            for i in _EQUITY_INDICES
        ],
        # ── Tier 2: intra-FI IG caps ─────────────────────────────────────
        *[
            {"type": "ineq", "fun": lambda w, i=i: ig_fi_cap - w[i]}
            for i in _IG_FI_INDICES
        ],
        # ── Tier 2: HYG quadratic risk-scaled cap ────────────────────────
        {"type": "ineq", "fun": lambda w: hyg_cap - w[_HYG_IDX]},
    ]
    return constraints


def optimize_portfolio(
    ann_mu: np.ndarray,
    ann_sig: np.ndarray,
    corr: np.ndarray,
    vol_ceiling: float,
    age: int,
    risk_score: float,
    rf: float = 0.04,
    n_restarts: int = 5,
) -> dict[str, float]:
    """
    Maximise Sharpe ratio subject to the two-tier constraint system.

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
    abs_caps = np.array([_ABS_CAPS.get(t, 0.40) for t in ETF_UNIVERSE])
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

    constraints = _build_constraints(
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
