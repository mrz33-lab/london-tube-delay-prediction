# London Tube Delay Prediction

**Machine Learning-Based Delay Severity Prediction for the London Underground Using Weather, Crowding, and Temporal Data**

> COMP1682 Final Year Individual Project — University of Greenwich
> Supervised by Tuan Vuong | [github.com/mrz33-lab/london-tube-delay-prediction](https://github.com/mrz33-lab/london-tube-delay-prediction)

![Version](https://img.shields.io/badge/version-2.0.0-blue) ![Status](https://img.shields.io/badge/status-active-blue) ![Python](https://img.shields.io/badge/python-3.14.2-blue) ![Tests](https://img.shields.io/badge/tests-154%20passed-brightgreen) ![Updated](https://img.shields.io/badge/updated-April%202026-lightgrey)

---

Predicts delay severity for 11 London Underground lines using 33 days of real TfL operational data merged with concurrent OpenWeatherMap observations. The **target variable is `delay_severity`** — an ordinal label (0 = Good Service, 1 = Minor Delays, 2 = Severe Delays) derived directly from the TfL Unified API status field. The pipeline trains three models — a naive persistence baseline, Ridge regression, and LightGBM — on 40 engineered input features (39 numeric + 1 categorical) with chronological train/test splits. LightGBM is tuned via Optuna TPE; XGBoost is also trained as a secondary comparison model when available. The system includes SHAP explainability, split-conformal calibration, a FastAPI backend, and a 7-tab Streamlit dashboard.

---

## Target Variable

The model predicts `delay_severity`, an ordinal encoding of the real TfL service status:

| Value | TfL Status |
|---|---|
| 0 | Good Service |
| 1 | Minor Delays |
| 2 | Severe Delays |

**`delay_minutes` is not the model target.** It is a synthetic proxy estimated from status labels at collection time and is excluded from all model features (`config.py` `exclude_columns`). It appears only in the dashboard UI for human-readable display. All reported metrics (MAE, RMSE, MAPE) are in **severity units** on the 0–2 ordinal scale, not minutes.

---

## Results

Evaluated on a held-out chronological test set (7,148 rows) from **run\_20260416\_135456** (seed 42).

| Model | Train MAE | Test MAE | Train RMSE | Test RMSE | Test R² | Test MAPE |
|---|---|---|---|---|---|---|
| Naive Baseline | 0.578 | 0.498 | 1.029 | 0.969 | −0.698 | 73.93% |
| Ridge Regression | 0.197 | 0.179 | 0.369 | 0.346 | 0.783 | 26.45% |
| **LightGBM** | **0.103** | **0.120** | **0.257** | **0.283** | **0.856** | **19.61%** |
| XGBoost (comparison) | 0.133 | 0.122 | 0.311 | 0.292 | 0.846 | 19.56% |

All metrics are in **severity units** (0–2 ordinal scale). Lower is better for MAE / RMSE / MAPE; higher is better for R². Source: `artifacts/run_20260416_135456/all_metrics.json`.

**LightGBM achieves a 75.9% reduction in MAE over the naive persistence baseline** ([0.4976 − 0.1200] / 0.4976 × 100; `all_metrics.json`).

| Metric | Value | Source |
|---|---|---|
| LightGBM Test MAE 95% CI | [0.105, 0.135] severity units | `all_metrics.json` |
| LightGBM Test RMSE 95% CI | [0.255, 0.310] severity units | `all_metrics.json` |
| Conformal calibration — median residual | 0.027 severity units | `run.log` |
| Conformal calibration — q95 | 0.689 severity units | `run.log` |

Bootstrap confidence intervals computed using block bootstrap (1,000 resamples) over the test set (seed 42).

### SHAP Feature Importance

Top 10 features by LightGBM SHAP importance (global mean |SHAP|), from `artifacts/run_20260416_135456/feature_importance.csv`:

| Rank | Feature | Importance |
|---|---|---|
| 1 | rolling_mean_delay_3 | 0.347 |
| 2 | lag_delay_1 | 0.108 |
| 3 | recent_disruption_rate | 0.067 |
| 4 | lag_delay_3 | 0.061 |
| 5 | rolling_std_delay_12 | 0.025 |
| 6 | rolling_mean_delay_12 | 0.021 |
| 7 | hour | 0.016 |
| 8 | day_of_week | 0.012 |
| 9 | network_avg_delay | 0.007 |
| 10 | hour_sin | 0.005 |

`rolling_mean_delay_3` dominates at approximately **3.2× the impact of the second-ranked feature** (`lag_delay_1`: 0.108). All top-4 features are lag or rolling delay statistics.

Weather features do not appear in the top-10 SHAP ranking despite ranking first in ablation analysis. This discrepancy arises from correlation mediation: weather conditions are partially captured by recent delay history, so their marginal SHAP contribution is lower than their independent ablation impact. This is a documented analytical finding, not a pipeline defect.

---

## Dataset

Real data collected 24/7 from two live APIs via a systemd service on an Oracle Cloud Free Tier VM:

| Property | Value | Source |
|---|---|---|
| Sources | TfL Unified API + OpenWeatherMap | `data_collection.py` |
| Collection window | 2026-03-10 00:21 → 2026-04-12 19:58 (~33 days) | `data_info.txt` |
| Raw rows | 35,738 | `data_info.txt` Shape: (35738, 15) |
| Lines covered | 11 London Underground lines | `config.yaml` tube\_lines |
| Raw CSV columns | 14 (`delay_severity` derived from status at load time) | CSV header |
| Columns after load | 15 | `data_info.txt` |
| Input features (pre-OHE) | 40 (39 numeric + 1 categorical: line) | `run.log` Training set: (28590, 40) |
| Preprocessed features (post-OHE) | 50 (one-hot encoding of line with 11 categories) | `explainability_summary.txt` |
| Train split | 28,590 rows (chronological) | `run.log` |
| Test split | 7,148 rows (chronological, no shuffling) | `run.log` |

**Class distribution** (full dataset, 35,738 rows):

| Class | Count | Share |
|---|---|---|
| 0 — Good Service | 27,252 | 76.3% |
| 2 — Severe Delays | 5,295 | 14.8% |
| 1 — Minor Delays | 3,191 | 8.9% |

The dataset is class-imbalanced: Good Service accounts for over three-quarters of observations. This is consistent with real TfL operational patterns and is acknowledged in §Limitations.

Data lives at `data/tfl_merged.csv`. When this file is present the pipeline uses real data automatically. The collector (`data_collection.py`) polls both APIs every 15 minutes (`COLLECTION_INTERVAL_SECONDS = 900`).

**Required environment variables:**
```
TFL_API_KEY=your_tfl_unified_api_key
WEATHER_API_KEY=your_openweathermap_api_key
```

**Crowding features** use static lookup tables (not real-time passenger counts). Ablation analysis confirms zero measured marginal impact from the crowding feature group alone, consistent with this approximation.

---

## Requirements

- Python 3.14.2
- LightGBM >=4.2.0 (4.6.0 was installed at the training run reported here)
- XGBoost >=3.2.0
- scikit-learn >=1.3

```bash
git clone https://github.com/mrz33-lab/london-tube-delay-prediction
cd london-tube-delay-prediction
py -m venv .venv
.venv\Scripts\activate
py -m pip install -r requirements.txt
```

---

## Model Pipeline

### Training order

1. **Naive persistence baseline** — predicts the previous observation's `delay_severity` unchanged. Serves as the performance floor.
2. **Ridge regression** — linear model with L2 regularisation. HPO via **GridSearchCV** (6 alpha values: 0.001, 0.01, 0.1, 1.0, 10.0, 100.0; 5-fold TimeSeriesSplit). Best alpha for the reported run: 100.0.
3. **LightGBM** — primary advanced model. HPO via **Optuna TPE sampler** (50 trials, 5-fold TimeSeriesSplit). Persistent Optuna study saved to `artifacts/optuna_study.db`.
4. **XGBoost** (when available) — trained in parallel as a secondary comparison model. HPO via **RandomizedSearchCV** (20 iterations, 5-fold TimeSeriesSplit).

### Fallback chain (LightGBM unavailable)

If LightGBM cannot be imported, `train_fallback_model` delegates to XGBoost. If XGBoost is also unavailable, it falls back to RandomForest (RandomizedSearchCV). In the current environment both LightGBM and XGBoost are available; the fallback path is not exercised.

### CV strategy

All models use 5-fold **TimeSeriesSplit** (`config.py` `cv_splits=5`). Folds grow chronologically; no future data is ever used to select hyperparameters.

LightGBM 5-fold CV MAE (from `cv_fold_scores.json`): mean = 0.138, std = 0.020.

---

## Feature Engineering

40 input features are derived from 11 raw columns (after excluding `timestamp`, `status`, `delay_minutes`, and the target `delay_severity`). After one-hot encoding of the `line` field (11 categories), the preprocessing pipeline produces **50 model-facing features**.

| Group | Features |
|---|---|
| Lag delays | `lag_delay_1`, `lag_delay_3` |
| Rolling mean | `rolling_mean_delay_3`, `rolling_mean_delay_12` |
| Rolling std (window ≥ 12 h only) | `rolling_std_delay_12` |
| Disruption rate | `recent_disruption_rate` |
| Weather delta | `temp_delta_1h`, `precipitation_delta_1h` |
| Interaction | `crowding_x_peak`, `precipitation_x_temp` |
| Network effects (leave-one-out) | `network_avg_delay`, `network_delay_volatility`, `lines_disrupted_ratio`, `is_network_wide_disruption` |
| Temporal encoding | `hour_sin`, `hour_cos`, `is_late_night`, `is_early_morning` |
| Topology (static) | `line_length_km`, `n_stations`, `n_interchange_stations`, `is_deep_tube`, `zone_coverage` |
| Train frequency (static) | `trains_per_hour`, `service_headway_min`, `capacity_pressure` |
| Event calendar | `is_major_event`, `event_crowd_boost`, `seasonal_demand_factor` |
| Raw weather / calendar | `temp_c`, `precipitation_mm`, `humidity`, `crowding_index`, `is_weekend`, `hour`, `day_of_week`, `month`, `peak_time`, `is_holiday` |
| Categorical (OHE) | `line` (11 categories → 11 binary columns) |

### Leakage protection

All lag and rolling features apply `shift(1)` before any rolling aggregation, computed per-line within each `groupby`. At training time, `_verify_no_leakage` (`features.py:302`) checks that the first observation in each per-line series is NaN for every lag/rolling feature. Violations are logged as `logger.warning`; the function does not raise an exception. No warnings were emitted during `run_20260416_135456`.

---

## Calibration

Split-conformal calibration is applied after training (`train.py:696`). The test set is partitioned 50/50 (chronologically): the first half (n = 3,574) serves as the calibration set; the second half is the evaluation set.

- **Median residual:** 0.027 severity units
- **q95:** 0.689 severity units

At inference time, `FutureDelayPredictor` uses these calibration scores to produce prediction intervals. Source: `artifacts/run_20260416_135456/run.log`.

---

## Usage

**1. Train all models**
```bash
py train.py
```
Runs the full pipeline: naive → Ridge → LightGBM (Optuna) → XGBoost → bootstrap CIs → conformal calibration → per-line residual quantiles → classification evaluation. Outputs a timestamped artifact directory:
```
artifacts/run_YYYYMMDD_HHMMSS/
  ├── best_model.pkl
  ├── all_metrics.json
  ├── feature_importance.csv
  ├── shap_*.png
  ├── classification/
  └── ...
```

**2. SHAP explainability**
```bash
py explain.py
```
Loads the latest artifact run and writes global SHAP summary plots, local waterfall plots, and a natural-language feature importance summary to the same artifact folder.

**3. Streamlit dashboard** (7 tabs)
```bash
py -m streamlit run app.py
```
Opens at `http://localhost:8501`. Tabs: Predictions · Performance · Line Comparison · Historical Trends · Data Collection · Risk Map · About.

**4. FastAPI backend**
```bash
py -m uvicorn api:app --reload --port 8000
```
Interactive docs at `http://localhost:8000/docs`.
Endpoints: `POST /predict`, `POST /predict/batch`, `POST /predict/forecast`, `GET /health`, `GET /lines`, `GET /`

> **Prototype mode:** If `data/tfl_merged.csv` is absent, `data.py` falls back to a synthetic dataset so the pipeline and dashboard remain runnable for demonstration purposes.

---

## Tests

```bash
py -m pytest tests/ -v                          # 154 passed
py -m pytest tests/ --cov=. --cov-report=html  # with HTML coverage report
```

| Category | Coverage |
|---|---|
| Schema validation | Column types, value ranges, null checks |
| Data leakage detection | Shift-before-roll ordering, no future data in features |
| Feature engineering | Per-line groupby correctness, lag/rolling outputs |
| API contract | Request/response shapes, error handling |

---

## Project Structure

```
├── config.py               # Central configuration (dataclasses)
├── data.py                 # Real / synthetic data loader; adds delay_severity from status
├── features.py             # Feature engineering (40 input → 50 preprocessed features),
│                           #   leakage check via _verify_no_leakage (warns, does not raise)
├── train.py                # Pipeline — naive, Ridge (GridSearchCV), LightGBM (Optuna TPE),
│                           #   XGBoost (RandomizedSearchCV), TimeSeriesSplit CV,
│                           #   block-bootstrap CIs, conformal calibration
├── explain.py              # SHAP global + local explainability, plots, summaries
├── future_prediction.py    # Inference for unseen inputs
├── api.py                  # FastAPI service (predict, batch, forecast, health, lines)
├── app.py                  # 7-tab Streamlit dashboard
├── line_metadata.py        # Static TfL line metadata (length, stations, interchanges)
├── data_collection.py      # Live TfL + weather collector (15-min polling)
├── tests/                  # 154 passing tests
├── analysis/               # Ablation study, ARIMA baseline, learning curves
├── data/
│   └── tfl_merged.csv      # 33-day real TfL + OpenWeatherMap dataset (35,738 rows)
└── artifacts/              # Per-run outputs — models, metrics, plots (gitignored)
```

---

## Methodology Summary

| Component | Detail | Source |
|---|---|---|
| Data collection | TfL Unified API + OpenWeatherMap, Oracle Cloud Free Tier VM, systemd | `data_collection.py` |
| Collection window | ~33 days (2026-03-10 to 2026-04-12), 15-min poll interval | `data_info.txt` |
| Dataset | 35,738 rows, 11 lines, 14 raw CSV columns | `data_info.txt` |
| Target variable | `delay_severity` — ordinal 0/1/2 (real TfL status label) | `config.py` FeatureConfig |
| Train / test split | Chronological — 28,590 train / 7,148 test (no shuffling) | `run.log` |
| Input features | 40 (pre-OHE); 50 after one-hot encoding of `line` | `run.log`, `explainability_summary.txt` |
| Models | Naive persistence, Ridge, LightGBM; XGBoost as comparison | `train.py` |
| HPO — Ridge | GridSearchCV, 6 alpha values, 5-fold TimeSeriesSplit | `train.py:224` |
| HPO — LightGBM | Optuna TPE, 50 trials, 5-fold TimeSeriesSplit | `train.py:302` |
| HPO — XGBoost | RandomizedSearchCV, 20 iterations, 5-fold TimeSeriesSplit | `train.py:359` |
| Evaluation | MAE, RMSE, R², MAPE (severity units), block-bootstrap 95% CI | `all_metrics.json` |
| Conformal calibration | Split-conformal, 50/50 test split, median=0.027, q95=0.689 severity units | `run.log` |
| Explainability | SHAP (global beeswarm + bar + local waterfall) + ablation study | `explain.py`, `analysis/` |
| Environment | Python 3.14.2, LightGBM >=4.2.0 (4.6.0 installed), XGBoost >=3.2.0, scikit-learn >=1.3 | `requirements.txt`, `model_info.json` |
| Artifact run | run\_20260416\_135456, seed 42 | `config.py` RANDOM\_SEED |

---

## Reproducibility

```bash
py -m pip install -r requirements.txt
py train.py     # deterministic at seed 42
py explain.py   # regenerates SHAP plots
```

Key pins: Python 3.14.2 · LightGBM >=4.2.0 · XGBoost >=3.2.0 · scikit-learn >=1.3 · optuna >=3.0.0 · seed 42.

Optuna stores trial history in `artifacts/optuna_study.db` (SQLite). Re-running `train.py` appends new trials to the existing study; delete the `.db` file for a clean run.

---

## Limitations

- **Ordinal target, not minute-level:** The model predicts `delay_severity` (0/1/2), not a continuous delay in minutes. The three-class granularity reflects what the TfL Unified API reliably exposes. `delay_minutes` is a synthetic proxy excluded from the model.

- **33-day collection window:** Data covers early March to mid-April 2026. Generalisation to seasonal variation (summer heat, winter ice, industrial action) is untested and cannot be claimed. This is acknowledged in the dissertation critical reflection as a scope limitation.

- **Severe class imbalance:** Good Service accounts for 76.3% of observations. Minor Delays (8.9%) is the hardest class to predict (ordinal logistic F1 = 0.42 for Minor Delays; `classification_results.json`). The regression formulation does not eliminate the underlying imbalance.

- **Static crowding data:** Crowding features are derived from fixed lookup tables rather than real-time passenger counts. Ablation analysis confirms zero measured marginal impact from the crowding feature group alone.

- **No station-level resolution:** Observations are aggregated at the line level. Localised station disruptions are not captured.

- **External shocks not modelled:** Strikes, planned engineering works, and signal failures are not represented as explicit features. They may appear indirectly through lag/rolling delay history.

- **Leakage check is advisory, not enforced:** `_verify_no_leakage` logs a warning on detection but does not raise. No warnings were observed in `run_20260416_135456`.

- **SHAP vs. ablation discrepancy:** Weather features rank first in ablation but not in SHAP. See §Results for the explanation.

---

## Citation

```
Marwan Amla. London Underground Delay Prediction.
COMP1682 Final Year Individual Project,
University of Greenwich, 2026.
https://github.com/mrz33-lab/london-tube-delay-prediction
```
