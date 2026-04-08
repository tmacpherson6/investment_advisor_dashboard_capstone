from flask import Flask, render_template, request, jsonify
import numpy as np
import io
import base64
import os
import json
import datetime
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

# Let's import some of our functions from logic.py, we may not use them all and can delete them later
from logic import (
    get_market_params,
    ETF_UNIVERSE,
    optimize_portfolio,
    compute_equity_ceiling,
)

# Create our Flask app
app = Flask(__name__)

# All market parameters — correlation, LSTM vol, and Sharpe-derived returns —
# are now computed in logic.py and exposed via get_market_params().
# ANN_MU is derived from LSTM vol + Sharpe priors. See logic.py for details.
mp = get_market_params()
CORR    = mp["corr"]
ANN_SIG = mp["ann_sig"]       # LSTM 30-trial avg vol (primary)
ANN_SIG_HIST = mp["ann_sig_hist"]  # Historical vol from yfinance (for viz)
ANN_MU  = mp["ann_mu"]        # rf + sharpe_prior * sigma_lstm

# Initialise feedback table (no-op if already exists or no DB configured)
with app.app_context():
    pass  # init_feedback_db() called below after routes are defined

# Risk-free rate — update to match current BIL yield if desired for more dynamic modelling. Must be consistent with logic.py for ANN_MU derivation.
RF_ANN = 0.04  

# Calculate trading days for different horizon options (used in simulation). 252 trading days per year is standard.
HORIZON_OPTIONS = {
    "1": 252,
    "2": 504,
    "5": 1260,
    "10": 2520,
    "20": 5040,
    "30": 7560,
}


# ─────────────────────────────────────────────
#  MOCK FUNCTIONS (Need To Repalce When SCF data is complete)
# ─────────────────────────────────────────────


def mock_risk_score(form_data):
    """Replace with: model.predict(features)"""
    age = int(form_data.get("age", 45))
    income = float(form_data.get("income", 100000))
    net_worth = float(form_data.get("net_worth", 300000))
    education = int(form_data.get("education", 3))
    experience = int(form_data.get("investment_experience", 2))

    score = 5.0
    score += 2 - (age - 30) / 20
    score += (income / 100000 - 1) * 0.8
    score += (experience - 2) * 0.5
    score += (education - 3) * 0.3
    score = float(np.clip(round(score, 1), 1, 10))

    # Contiguous ranges using < on upper bound — prevents StopIteration on half-point scores
    # Replace thresholds with calibrated values once real model is trained
    if score <= 2.0:
        vol_ceiling = 0.05
    elif score <= 4.0:
        vol_ceiling = 0.08
    elif score <= 6.0:
        vol_ceiling = 0.13
    elif score <= 8.0:
        vol_ceiling = 0.18
    else:
        vol_ceiling = 0.24
    return {"score": score, "vol_ceiling": vol_ceiling}


# ─────────────────────────────────────────────
#  SIMULATIONS
# ─────────────────────────────────────────────


def run_simulation(
    weights, horizon_days=252, n_paths=1000, start_value=100_000, seed=42
):
    """
    Simulates portfolio paths over the given horizon.
    ANN_MU / ANN_SIG / CORR are annualised — daily scaling handles any horizon.
    Replace with real forward-looking estimates.
    """
    w = np.array(list(weights.values()))
    daily_mu = ANN_MU / 252
    daily_sig = ANN_SIG / np.sqrt(252)
    daily_cov = np.diag(daily_sig) @ CORR @ np.diag(daily_sig)

    rng = np.random.default_rng(seed)
    paths = np.zeros((n_paths, horizon_days + 1))
    paths[:, 0] = start_value
    daily_rets = np.zeros((n_paths, horizon_days))

    for t in range(1, horizon_days + 1):
        shocks = rng.multivariate_normal(daily_mu, daily_cov, size=n_paths)
        port_ret = shocks @ w
        daily_rets[:, t - 1] = port_ret
        paths[:, t] = paths[:, t - 1] * (1 + port_ret)

    return paths, daily_rets


def compute_mc_stats(paths, daily_rets, start_value, vol_ceiling, horizon_years):
    """
    Derives statistics from simulation paths.
    - Return & vol: median ± 1 STD of the cross-path distribution
    - Sharpe: median ± 1 STD
    - Terminal value & drawdown: kept as percentiles (more intuitive)
    """
    horizon_days = paths.shape[1] - 1
    ann_factor = 252 / horizon_days
    terminal = paths[:, -1]

    # ── Per-path annualised return ──
    path_returns = ((terminal / start_value) ** ann_factor - 1) * 100
    ret_mean = float(np.mean(path_returns))
    ret_std = float(np.std(path_returns))
    ret_median = float(np.median(path_returns))

    # ── Per-path annualised volatility ──
    path_vols = daily_rets.std(axis=1) * np.sqrt(252) * 100
    vol_mean = float(np.mean(path_vols))
    vol_std = float(np.std(path_vols))
    vol_median = float(np.median(path_vols))

    # ── Per-path Sharpe ──
    path_ret_raw = (terminal / start_value) ** ann_factor - 1
    path_vol_raw = daily_rets.std(axis=1) * np.sqrt(252)
    path_sharpes = np.where(
        path_vol_raw > 0, (path_ret_raw - RF_ANN) / path_vol_raw, np.nan
    )
    sharpe_mean = float(np.nanmean(path_sharpes))
    sharpe_std = float(np.nanstd(path_sharpes))
    sharpe_median = float(np.nanmedian(path_sharpes))

    # ── Terminal value percentiles ──
    tv_p10, tv_p50, tv_p90 = np.percentile(terminal, [10, 50, 90])

    # ── Max drawdown percentiles ──
    cummax = np.maximum.accumulate(paths, axis=1)
    max_dds = (paths / cummax - 1).min(axis=1) * 100
    dd_p50, dd_p10 = np.percentile(max_dds, [50, 10])  # p10 = worst 10%

    # ── Constraint status: green / amber (within 1%) / red (over ceiling) ──
    vol_median_dec = vol_median / 100
    if vol_median_dec <= vol_ceiling:
        constraint_status = "green"
    elif vol_median_dec <= vol_ceiling + 0.01:
        constraint_status = "amber"
    else:
        constraint_status = "red"

    def r(x, d=1):
        return round(float(x), d)

    return {
        # Annualised return — median ± 1 STD
        "ret_median": r(ret_median),
        "ret_lo": r(ret_mean - ret_std),  # ~16th pct
        "ret_hi": r(ret_mean + ret_std),  # ~84th pct
        # Annualised vol — median ± 1 STD
        "vol_median": r(vol_median),
        "vol_lo": r(max(0.0, vol_mean - vol_std)),
        "vol_hi": r(vol_mean + vol_std),
        # Sharpe — median ± 1 STD
        "sharpe_median": r(sharpe_median, 2),
        "sharpe_lo": r(sharpe_mean - sharpe_std, 2),
        "sharpe_hi": r(sharpe_mean + sharpe_std, 2),
        # Terminal value (percentiles — kept, more intuitive for dollar amounts)
        "tv_p10": round(float(tv_p10), 0),
        "tv_p50": round(float(tv_p50), 0),
        "tv_p90": round(float(tv_p90), 0),
        # Max drawdown (percentiles — kept)
        "dd_p50": r(dd_p50),
        "dd_p10": r(dd_p10),
        # Meta
        "vol_ceiling_ann": r(vol_ceiling * 100),
        "constraint_status": constraint_status,
        "horizon_years": horizon_years,
    }


def generate_gif(paths, start_value, horizon_years):
    """Animated spaghetti-plot GIF. Subsamples to ~60 frames."""
    n_paths, T = paths.shape
    horizon_days = T - 1

    BG = "#0D1117"
    GOLD = "#C9A84C"
    LIGHT = "#E8E0D0"
    GREY = "#2A2F3A"

    frames_buf = []
    step = max(1, n_paths // 60)
    x_label = f"Trading Days  ({horizon_years}yr horizon)"

    for end in range(1, n_paths + 1, step):
        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=90)
        fig.patch.set_facecolor(BG)
        ax.set_facecolor(BG)

        for i in range(end):
            ax.plot(paths[i], lw=0.5, alpha=0.3, color=GREY)

        ax.plot(paths[end - 1], lw=1.8, color=GOLD, zorder=5)

        days = np.arange(horizon_days + 1)
        p5_line = np.percentile(paths[:end], 5, axis=0)
        p95_line = np.percentile(paths[:end], 95, axis=0)
        p50_line = np.percentile(paths[:end], 50, axis=0)
        ax.fill_between(days, p5_line, p95_line, alpha=0.08, color=GOLD)
        ax.plot(days, p50_line, lw=1.2, color=GOLD, linestyle="--", alpha=0.7)

        ax.set_xlabel(x_label, color=LIGHT, fontsize=9)
        ax.set_ylabel("Portfolio Value ($)", color=LIGHT, fontsize=9)
        ax.set_title(
            f"Monte Carlo Simulation  |  {end} / 1,000 paths  |  {horizon_years}yr",
            color=LIGHT,
            fontsize=11,
            pad=10,
        )
        ax.tick_params(colors=LIGHT, labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor(GREY)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

        stats_text = f"P5: ${p5_line[-1]:,.0f}   Median: ${p50_line[-1]:,.0f}   P95: ${p95_line[-1]:,.0f}"
        ax.text(
            0.02,
            0.96,
            stats_text,
            transform=ax.transAxes,
            color=LIGHT,
            fontsize=7.5,
            alpha=0.85,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=GREY, edgecolor="none"),
        )

        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", facecolor=BG)
        plt.close(fig)
        buf.seek(0)
        frames_buf.append(buf.read())

    pil_frames = [Image.open(io.BytesIO(f)).convert("RGBA") for f in frames_buf]
    gif_buf = io.BytesIO()
    pil_frames[0].save(
        gif_buf,
        format="GIF",
        save_all=True,
        append_images=pil_frames[1:],
        loop=0,
        duration=150,
    )
    gif_buf.seek(0)
    return base64.b64encode(gif_buf.read()).decode("utf-8")


def build_results(weights, vol_ceiling, start_value=100_000, horizon_years=1):
    """Single entry point — runs simulation once, derives all stats and GIF."""
    horizon_days = HORIZON_OPTIONS.get(str(horizon_years), 252)
    paths, daily_rets = run_simulation(
        weights, horizon_days=horizon_days, start_value=start_value
    )
    stats = compute_mc_stats(paths, daily_rets, start_value, vol_ceiling, horizon_years)
    gif_b64 = generate_gif(paths, start_value, horizon_years)
    return stats, gif_b64


# ─────────────────────────────────────────────
#  ROUTES
# ─────────────────────────────────────────────


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    form_data = request.form.to_dict()
    start_value = float(form_data.get("investment_amount", 100_000))
    horizon_years = int(form_data.get("horizon_years", 1))
    age = int(form_data.get("age", 45))

    risk = mock_risk_score(form_data)
    weights = optimize_portfolio(
        ann_mu=ANN_MU,
        ann_sig=ANN_SIG,
        corr=CORR,
        vol_ceiling=risk["vol_ceiling"],
        age=age,
        risk_score=risk["score"],
        rf=RF_ANN,
    )
    stats, gif = build_results(weights, risk["vol_ceiling"], start_value, horizon_years)

    return render_template(
        "results.html",
        form_data=form_data,
        risk=risk,
        weights=weights,
        stats=stats,
        etfs=ETF_UNIVERSE,
        gif_b64=gif,
        horizon_years=horizon_years,
        # Vol comparison data for visualization
        lstm_vol={etf: round(float(ANN_SIG[i]*100), 2) for i, etf in enumerate(ETF_UNIVERSE)},
        hist_vol={etf: round(float(ANN_SIG_HIST[i]*100), 2) for i, etf in enumerate(ETF_UNIVERSE)},
        ann_mu={etf: round(float(ANN_MU[i]*100), 2) for i, etf in enumerate(ETF_UNIVERSE)},
    )


# ─────────────────────────────────────────────
#  FEEDBACK — Schema + Route with help of GENAI (stores to Postgres if DATABASE_URL configured, else JSONL fallback)
# ─────────────────────────────────────────────

FEEDBACK_DDL = """
CREATE TABLE IF NOT EXISTS risk_feedback (
    id               SERIAL PRIMARY KEY,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Model output
    model_score      NUMERIC(4,1) NOT NULL,
    override_score   NUMERIC(4,1),
    preferred_score  NUMERIC(4,1) NOT NULL,
    direction        VARCHAR(10)  NOT NULL CHECK (direction IN ('too_high','too_low')),
    notes            TEXT,

    -- Feature snapshot for retraining (mirrors mock_risk_score inputs)
    age              SMALLINT,
    income           NUMERIC(14,2),
    net_worth        NUMERIC(14,2),
    education        SMALLINT,
    investment_experience SMALLINT,

    -- Signed error: positive => model over-predicted, negative => under-predicted
    score_delta      NUMERIC(4,1) GENERATED ALWAYS AS (model_score - preferred_score) STORED
);
CREATE INDEX IF NOT EXISTS idx_rf_created   ON risk_feedback (created_at);
CREATE INDEX IF NOT EXISTS idx_rf_direction ON risk_feedback (direction);
"""


def get_db_conn():
    """Return a psycopg2 connection from DATABASE_URL. Returns None if unavailable."""
    try:
        import psycopg2

        db_url = os.environ.get("DATABASE_URL", "")
        if not db_url:
            return None
        if db_url.startswith("postgres://"):
            db_url = db_url.replace("postgres://", "postgresql://", 1)
        return psycopg2.connect(db_url)
    except Exception:
        return None


def init_feedback_db():
    """Idempotently create the risk_feedback table. Call once at startup."""
    conn = get_db_conn()
    if conn is None:
        app.logger.warning(
            "feedback_db: no DATABASE_URL — feedback falls back to JSONL log."
        )
        return
    with conn:
        with conn.cursor() as cur:
            cur.execute(FEEDBACK_DDL)
    conn.close()


FEEDBACK_LOG = os.path.join(os.path.dirname(__file__), "feedback_log.jsonl")


@app.route("/api/feedback", methods=["POST"])
def api_feedback():
    data = request.get_json(silent=True) or {}

    for field in ("model_score", "preferred_score", "direction"):
        if field not in data:
            return jsonify({"ok": False, "error": f"Missing field: {field}"}), 400
    if data["direction"] not in ("too_high", "too_low"):
        return jsonify(
            {"ok": False, "error": "direction must be 'too_high' or 'too_low'"}
        ), 400

    record = {
        "model_score": float(data["model_score"]),
        "override_score": float(data["override_score"])
        if data.get("override_score") is not None
        else None,
        "preferred_score": float(data["preferred_score"]),
        "direction": data["direction"],
        "notes": str(data["notes"])[:1000] if data.get("notes") else None,
        "age": int(data["age"]) if data.get("age") is not None else None,
        "income": float(data["income"]) if data.get("income") is not None else None,
        "net_worth": float(data["net_worth"])
        if data.get("net_worth") is not None
        else None,
        "education": int(data["education"])
        if data.get("education") is not None
        else None,
        "investment_experience": int(data["investment_experience"])
        if data.get("investment_experience") is not None
        else None,
        "score_delta": round(
            float(data["model_score"]) - float(data["preferred_score"]), 1
        ),
    }

    conn = get_db_conn()
    if conn:
        try:
            with conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """INSERT INTO risk_feedback
                             (model_score, override_score, preferred_score, direction, notes,
                              age, income, net_worth, education, investment_experience)
                           VALUES
                             (%(model_score)s, %(override_score)s, %(preferred_score)s,
                              %(direction)s, %(notes)s, %(age)s, %(income)s, %(net_worth)s,
                              %(education)s, %(investment_experience)s)""",
                        record,
                    )
            conn.close()
            return jsonify({"ok": True, "storage": "postgres"})
        except Exception as e:
            app.logger.error(f"Feedback DB write failed: {e}")
            conn.close()

    # JSONL fallback (dev / no DB configured)
    try:
        record["created_at"] = datetime.datetime.utcnow().isoformat()
        with open(FEEDBACK_LOG, "a") as f:
            f.write(json.dumps(record) + "\n")
        return jsonify({"ok": True, "storage": "jsonl_fallback"})
    except Exception as e:
        app.logger.error(f"Feedback JSONL fallback failed: {e}")
        return jsonify({"ok": False, "error": "Storage unavailable"}), 500


@app.route("/api/recalculate", methods=["POST"])
def api_recalculate():
    try:
        data = request.get_json()
        app.logger.info(f"recalculate payload: {data}") 
        vol_ceiling   = float(data.get("vol_ceiling", 0.13))
        start_value   = float(data.get("start_value", 100_000))
        horizon_years = int(data.get("horizon_years", 1))
        weights = optimize_portfolio(
            ann_mu=ANN_MU, ann_sig=ANN_SIG, corr=CORR,
            vol_ceiling=vol_ceiling,
            age=int(data.get("age", 45)),
            risk_score=float(data.get("risk_score", 5.0)),
            rf=RF_ANN,
        )
        stats, gif = build_results(weights, vol_ceiling, start_value, horizon_years)
        return jsonify({"weights": weights, "stats": stats, "gif_b64": gif})
    except Exception as e:
        app.logger.exception("api_recalculate failed")    
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    init_feedback_db()
    app.run(debug=True)