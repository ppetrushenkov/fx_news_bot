# FX Volatility Alert Bot

A Telegram bot that warns FX traders **~1 hour before scheduled macro news events** when the models expect elevated volatility — wide ranges, multiple direction reversals, or a swing-failure (false breakout) pattern. Built end-to-end: data collection, feature engineering, model training, and a production Telegram bot.

> The idea: before high-impact news (NFP, rate decisions, CPI, etc.) the market often either makes a large, choppy move in both directions or fakes a breakout and reverses. Knowing this in advance means "don't trade this hour" or "trade the range carefully, expect fakeouts."

## What it does

- Pulls the economic calendar (news events, importance, forecast/previous/actual) from **TradingView**.
- Pulls OHLC price data for major USD pairs from **Twelve Data**.
- An hour before each scheduled news event (or combination of events), a scheduled job builds live features and runs them through a set of CatBoost models to forecast:
  - **Expected range** for the next 1h / 3h / 6h / 24h (quantile regression, p10/p50/p90)
  - **Number of direction changes** (swings) expected — how "choppy" the move will be (ordinal regression)
  - **Chaos probability** — likelihood the move exceeds the pair's own ATR (binary classification, per horizon)
  - **Trend vs. flat regime** for the next 1–2 days (binary/multiclass)
  - **Swing Failure Pattern (SFP)** probability — a false breakout of a recent high/low that reverses hard, which is common around certain news events (binary classification)
- Sends a Telegram alert summarizing which pairs are at risk, the expected regime, and any detected noise (chaos / range expansion / spikes / SFP) — so the user can decide whether to sit the news out, trade it cautiously, or look for a range-trade setup.
- Also supports daily/weekly calendar digests, per-user importance filters, and timezone-aware delivery.

## Why it's interesting

- Full ML lifecycle, not a notebook demo: SQL-backed feature store → offline training with walk-forward (time-series) cross-validation → calibrated probabilities → saved model artifacts → live inference inside a scheduled production job.
- Multiple target types trained on the same feature set with CatBoost: quantile regression, ordinal regression, binary and multiclass classification.
- Class-imbalance handling and probability calibration (isotonic/Platt) tuned for an alerting use case, where recall matters more than precision (missing a volatility spike is worse than a false alarm).
- Experiment tracking with MLflow across dozens of training runs per target/horizon.
- A real async Telegram bot (aiogram 3.x) with FSM-based onboarding, a job scheduler that re-plans itself daily around the next day's news calendar, and a normalized SQLAlchemy schema (events, prices, features, user settings).

## Architecture

```
News (TradingView)  ─┐
                      ├──▶  Feature pipeline  ──▶  CatBoost models  ──▶  chaos_score / SFP / range / regime
Prices (Twelve Data) ─┘             │
                                    ▼
                          SQLAlchemy DB (events, prices, features)
                                    │
                                    ▼
                     APScheduler jobs (weekly recalculation of
                     "1h before next event" triggers per pair)
                                    │
                                    ▼
                       aiogram Telegram bot ──▶  user alerts
```

## Tech stack

| Layer | Tools |
|---|---|
| Data sources | TradingView (economic calendar), Twelve Data (OHLC prices) |
| Storage | SQLAlchemy ORM + SQL database (events, prices, engineered features, user settings) |
| Feature engineering | pandas, custom `sklearn`-style transformers (event aggregation, price/target featurizers, lagged rolling stats) |
| Modeling | CatBoost (binary, multiclass, ordinal, multi-quantile regression), scikit-learn, walk-forward CV, isotonic/Platt calibration |
| Experiment tracking | MLflow |
| Bot | aiogram 3.x (async, FSM), APScheduler (cron + one-off jobs) |
| Language | Python (async/await throughout) |

## Modeling approach

Data is aggregated to an hourly grid per currency pair, joined with a rolling window of nearby news events (importance, category, source, time-to-event), and enriched with lagged rolling statistics of past outcomes for the same event type. Roughly a dozen targets are trained per experiment, all on the same feature matrix, each solving a different piece of "what will the market do in the next N hours":

- `trg_future_range_{1,3,6,24}h` — quantile regression (p10/p50/p90) on the expected high-low range
- `trg_is_chaos_{1,3,6,24}h` — binary classification: will the range exceed the pair's own ATR
- `trg_dir_changes` — ordinal regression: how many times price reverses direction (swing count)
- `trg_direction_{1,2}day` — multiclass: trend vs. flat regime
- `trg_is_sfp` — binary classification: swing failure pattern (false breakout that reverses)
- `trg_big_doji` / `trg_is_extremum_breakout` — binary classification: spike / range-extremum breakout detection

Validation uses **expanding-window, time-based splits** (no shuffling — this is time series) with quarter-based folds, and results are logged to MLflow per fold. Example in-sample metrics from the notebook:

- Chaos prediction (24h horizon): **Accuracy 0.887, AUC 0.945**
- Trend/flat regime: **Accuracy 0.895, AUC 0.962**
- Range prediction (24h, ordinal-style range bucket): MAE 0.74, Spearman 0.57

*(These are training-run snapshots from experimentation, not held-out production performance — see [Limitations](#limitations--next-steps).)*

## Repository structure

```
.
├── app.py                          # Bot entrypoint: handlers, scheduler jobs, alert composition
├── config.py                       # Configuration / secrets loading
├── train_and_retrain_model.ipynb   # End-to-end training notebook: data → features → CatBoost models
├── training_functions.py           # Walk-forward CV training helpers (quantile / classification / ordinal)
├── bot/
│   ├── scheduler.py                 # Computes "next event − 1h" trigger times, weekly recalculation job
│   └── feature_engineer.py          # Time helpers for scheduling
├── ml/
│   ├── predictor.py                 # Loads trained models, runs live inference (FxRangePredictor)
│   ├── news_featuring.py            # News event feature pipeline (aggregation, rounding, one-hot)
│   ├── price_featuring.py           # Price/technical feature pipeline
│   ├── preprocessing.py             # sklearn-style transformers used in the notebook and in production
│   ├── train_catboost_unified.py    # Shared CatBoost training utilities
│   └── train_catboost_handle_disbalance.py  # Imbalance handling + calibration for alert-oriented targets
├── db/
│   ├── models.py                    # SQLAlchemy models: Events, Prices, UserSettings, ...
│   ├── database.py                  # Engine/session setup
│   └── data_handler.py              # DB read/write and calendar refresh logic
└── models/
    └── feature_transformers/        # Saved (joblib) fitted transformers used at inference time
```

## How it runs

1. On startup, tables are created (SQLAlchemy) if they don't exist yet: events, prices, engineered features, user settings.
2. A **weekly job** (Sunday 18:00 UTC) refreshes the upcoming news calendar and price history.
3. A **daily job** (23:59 UTC) rebuilds the next day's schedule: for every pair/event combination, it schedules a one-off job at *event time − 1 hour* that will build live features and run the models.
4. When that job fires, the bot builds the current feature snapshot, runs it through the CatBoost models, computes range forecasts / chaos flags / SFP probability / regime, and pushes a formatted alert to subscribed users — filtered by their importance and timezone preferences.
5. Users can also request daily/weekly calendar digests and a manual "check the market now" forecast at any time.

## Telegram commands

```
/start              — onboarding
/set_alerts         — toggle daily / weekly / volatility alerts
/set_importance     — filter news by importance (low/medium/high)
/set_gmt             — set timezone for alert delivery
/today_summary       — today's news events
/tomorrow_summary    — tomorrow's news events
/weekly_summary      — this week's news events
/test_check_the_market — run a live forecast on demand
```

## Setup

> This is a personal/portfolio project — some config (API keys, DB connection) is environment-specific and not included here.

```bash
git clone <repo-url>
cd <repo>
pip install -r requirements.txt

# set environment variables (Telegram bot token, Twelve Data API key, DB URL, etc.)
cp .env.example .env

python app.py
```

Model training/retraining is done separately in `train_and_retrain_model.ipynb`, which walks through: loading events and prices from the database, building the feature pipeline, and training/evaluating each CatBoost target with walk-forward cross-validation and MLflow tracking.

## Limitations & next steps

- Metrics quoted above are from training-time evaluation; a held-out, chronologically-later test period and live paper-trading track record would make the case stronger.
- No backtested P&L / strategy layer yet — the bot currently informs "expect volatility," not "take this trade."
- Feature/model set is being iterated on (see the `V2` section of the notebook); next steps include tightening calibration for the SFP model and adding a lightweight backtest harness.

## Disclaimer

This project is for educational and research purposes. It is not financial advice, and volatility forecasts are probabilistic, not guarantees.
