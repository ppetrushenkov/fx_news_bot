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
├── main.py                          # Main entrypoint point (create database, run bot)
├── config.py                        # Configuration / secrets loading
├── training_functions.py            # Walk-forward CV training helpers (quantile / classification / ordinal)
├── bot/
    ├── app.py                       # Bot entrypoint: handlers, scheduler jobs, alert composition
│   ├── scheduler.py                 # Computes "next event − 1h" trigger times, weekly recalculation job
│   └── feature_engineer.py          # Time helpers for scheduling
├── ml/
│   ├── predictor.py                 # Loads trained models, runs live inference (FxRangePredictor)
│   ├── news_featuring.py            # News event feature pipeline (aggregation, rounding, one-hot)
│   ├── price_featuring.py           # Price/technical feature pipeline
│   └── preprocessing.py             # sklearn-style transformers used in the notebook and in production
├── db/
│   ├── models.py                    # SQLAlchemy models: Events, Prices, UserSettings, ...
│   ├── database.py                  # Engine/session setup
│   └── data_handler.py              # DB read/write and calendar refresh logic
├── models/                          # (In .gitignore)
     ├── ml/                         # Saved (joblib) CatBoost models
│    └── feature_transformers/       # Saved (joblib) fitted transformers used at inference time
└── notebooks/
     └── train_and_retrain_model.ipynb   # End-to-end training notebook: data → features → CatBoost models
```

## How it runs

1. On startup, tables are created (SQLAlchemy) if they don't exist yet: events, prices, engineered features, user settings.
2. A **weekly job** (Sunday 18:00 UTC) refreshes the upcoming news calendar and price history.
3. A **daily job** (23:59 UTC) rebuilds the next day's schedule: for every pair/event combination, it schedules a one-off job at *event time − 1 hour* that will build live features and run the models.
4. When that job fires, the bot builds the current feature snapshot, runs it through the CatBoost models, computes range forecasts / chaos flags / SFP probability / regime, and pushes a formatted alert to subscribed users — filtered by their importance and timezone preferences.
5. Users can also request daily/weekly calendar digests and a manual "check the market now" forecast at any time.

## Telegram commands

```
/start               — onboarding
/set_alerts          — toggle daily / weekly / volatility alerts
/set_importance      — filter news by importance (low/medium/high)
/set_gmt             — set timezone for alert delivery
/today_summary       — today's news events
/tomorrow_summary    — tomorrow's news events
/weekly_summary      — this week's news events
/check_the_market    — run a live forecast on demand
```

### Start:
Command: /start

![start.png](imgs/start.png)

### Main keyboard (available after "start" command)
![main_keyboard.png](imgs/main_keyboard.png)

### Help (/help)
![help.png](imgs/help.png)

### Set GMT (/set_gmt)
You can set your local GMT to show events corresponding to your local time. 
![set_timezone.png](imgs/set_timezone.png)

### Set event importance to show (/set_importance)
You can set the importance of events, if you want to see other, less important events.

![importance_settings.png](imgs/importance_settings.png)

### Events summary (example for "today_summary" here)
Shows the events summary for today (tomorrow / on the week).

![today_summary.png](imgs/today_summary.png)

### Forecast (/check_the_market)
The button "Make a forecast" runs all fitted models, that were trained before and shows forecast 
for all supported tickers.

Predictions to show:
- Future range for 1 hour might be greater 4 ATR
- Future range for 24 hours might be greater it's daily ATR
- Regime: Shows Trend / Flat for ticker and the value of future predicted swings
- Noise:
  - Swing Failure Pattern (False break out and the price goes in the different direction)
  - Double extremum breakout (The price might break out in the channel (21 period) from both sides)
  - Big Spike (The big bar with a big wick)
  - Chaos (If the sum of separate ranges are way greater than the total one)

![do_predictions.png](imgs/do_predictions.png)

---

## Metrics
### Total Future Range
| Metric             | Horizon 1h | Horizon 3h | Horizon 6h | Horizon 24h |
|:-------------------| :--- | :--- | :--- | :--- |
| **pinball q0.1**   | 0.06251665479566325 | 0.10391680618080708 | 0.12360434035932567 | 0.1570744735573364 |
| **pinball q0.5**   | 0.1873629355071192 | 0.2730939715008943 | 0.31498242235620993 | 0.3597971454383668 |
| **pinball q0.9**   | 0.11177433379631857 | 0.13813715833913462 | 0.18221943389014125 | 0.249675241182079 |
| **coverage q0.1**  | 0.1057542768273717 | 0.12716832156956573 | 0.15372652231128126 | 0.14295968417274793 |
| **coverage q0.5**  | 0.4952745543725326 | 0.5201579136260318 | 0.5444431152051681 | 0.537983012322048 |
| **coverage q0.9**  | 0.8902978825218327 | 0.8798899389879172 | 0.9090800334968298 | 0.9045340351716713 |
| **interval width** | 1.085288738388673 | 1.6226423648402406 | 1.9075807472026038 | 2.1405049936391047 |


### Direction Changes
| Metric | Value |
| :--- | :--- |
| QWK (Quadratic Weighted Kappa) | 0.4135 |
| MAE (Rounded Predictions) | 1.0498 |
| MAE (Continuous Predictions) | 1.0722 |
| RMSE (Rounded Predictions) | 1.3761 |
| Exact Match Accuracy | 0.2801 |
| Within-1 Accuracy | 0.7442 |


### Regime (Trend / Flat / None)
#### 1 Day:
![regime_1_day_overall_confusion_matrix.png](imgs/regime_1_day_overall_confusion_matrix.png)

| Class Label | Precision | Recall | F1-Score | Support |
| :---        | :---      | :---   | :---     |    :--- |
| Flat | 0.5849412671 | 0.8187601078 | 0.6823767269 | 18550 |
| None | 0.8684860459 | 0.5936022957 | 0.7052040509 | 47392 |
| Trend | 0.4147376543 | 0.8095895569 | 0.548492708 | 7967 |

#### 2 Days:
![regime_2_days_overall_confusion_matrix.png](imgs/regime_2_days_overall_confusion_matrix.png)

| Class Label | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| Flat | 0.5018034145 | 0.6274051309 | 0.5576189416 | 29936 |
| None | 0.5051109617 | 0.2496758967 | 0.3341712455 | 30083 |
| Trend | 0.3266543267 | 0.5082073434 | 0.3976901408 | 13890 |

### Noise targets
### Swing Failure Pattern

| Name | Threshold | PR-AUC | ROC-AUC | Brier Score | Log Loss | F0.5 | F1 | F2 | F3 | Precision | Recall | Base Rate |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| best_f0.5 | 0.8747533895 | 0.3810180706 | 0.8755308942 | 0.1408329453 | 0.4258222505 | 0.4471371277 | 0.3526244953 | 0.2910949392 | 0.2750944981 | 0.5444155844 | 0.2607613834 | 0.0543776807 |
| best_f1 | 0.7818015112 | 0.3810180706 | 0.8755308942 | 0.1408329453 | 0.4258222505 | 0.385959005 | 0.4096493187 | 0.4364380563 | 0.4461635771 | 0.3716312057 | 0.456332421 | 0.0543776807 |
| best_f2 | 0.629072507 | 0.3810180706 | 0.8755308942 | 0.1408329453 | 0.4258222505 | 0.2735503951 | 0.351924077 | 0.4932400423 | 0.5694631437 | 0.2381874175 | 0.6735506345 | 0.0543776807 |
| best_f3 | 0.5012070017 | 0.3810180706 | 0.8755308942 | 0.1408329453 | 0.4258222505 | 0.2111920677 | 0.2918738136 | 0.4723107978 | 0.5949003279 | 0.1783288231 | 0.80343369 | 0.0543776807 |


### Double Extremum Breakout

| Name | Threshold | PR-AUC | ROC-AUC | Brier Score | Log Loss | F0.5 | F1 | F2 | F3 | Precision | Recall | Base Rate |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| best_f0.5 | 0.847585 | 0.258275 | 0.899663 | 0.1252 | 0.375324 | 0.323394 | 0.329060 | 0.334928 | 0.336931 | 0.319723 | 0.338958 | 0.036883 |
| best_f1 | 0.803244 | 0.258275 | 0.899663 | 0.1252 | 0.375324 | 0.305237 | 0.348593 | 0.406305 | 0.430037 | 0.281866 | 0.456713 | 0.036883 |
| best_f2 | 0.695059 | 0.258275 | 0.899663 | 0.1252 | 0.375324 | 0.245248 | 0.320594 | 0.462767 | 0.543040 | 0.212028 | 0.657007 | 0.036883 |
| best_f3 | 0.583728 | 0.258275 | 0.899663 | 0.1252 | 0.375324 | 0.199692 | 0.276993 | 0.451939 | 0.572459 | 0.168368 | 0.780631 | 0.036883 |


### Big Spike

| Name | Threshold | PR-AUC | ROC-AUC | Brier Score | Log Loss | F0.5 | F1 | F2 | F3 | Precision | Recall | Base Rate |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| best_f0.5 | 0.653579 | 0.329347 | 0.736786 | 0.214357 | 0.609275 | 0.355251 | 0.362060 | 0.369135 | 0.371555 | 0.350853 | 0.374007 | 0.153351 |
| best_f1 | 0.504404 | 0.329347 | 0.736786 | 0.214357 | 0.609275 | 0.310135 | 0.392176 | 0.533235 | 0.605876 | 0.272176 | 0.701429 | 0.153351 |
| best_f2 | 0.390373 | 0.329347 | 0.736786 | 0.214357 | 0.609275 | 0.248067 | 0.340212 | 0.541267 | 0.674049 | 0.210125 | 0.893153 | 0.153351 |
| best_f3 | 0.312989 | 0.329347 | 0.736786 | 0.214357 | 0.609275 | 0.231902 | 0.323462 | 0.534495 | 0.683036 | 0.195087 | 0.945915 | 0.153351 |


### Chaos (horizon 24 hour)

| Name | Threshold | PR-AUC | ROC-AUC | Brier Score | Log Loss | F0.5 | F1 | F2 | F3 | Precision | Recall | Base Rate |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| best_f0.5 | 0.811133 | 0.478817 | 0.854502 | 0.164369 | 0.480837 | 0.496459 | 0.433001 | 0.383928 | 0.369952 | 0.550216 | 0.356957 | 0.130201 |
| best_f1 | 0.697565 | 0.478817 | 0.854502 | 0.164369 | 0.480837 | 0.453062 | 0.492768 | 0.540102 | 0.557967 | 0.429965 | 0.577055 | 0.130201 |
| best_f2 | 0.482814 | 0.478817 | 0.854502 | 0.164369 | 0.480837 | 0.345395 | 0.442261 | 0.614636 | 0.706412 | 0.301388 | 0.830406 | 0.130201 |
| best_f3 | 0.309924 | 0.478817 | 0.854502 | 0.164369 | 0.480837 | 0.283261 | 0.383313 | 0.592644 | 0.724535 | 0.241276 | 0.931934 | 0.130201 |

---
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
