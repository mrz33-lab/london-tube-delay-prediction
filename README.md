# 🚇 London Tube Delay Prediction

**Machine Learning-Based Delay Prediction for the London Underground Using Weather, Crowding, and Temporal Data**

> COMP1682 Final Year Individual Project — University of Greenwich  
> Supervised by Tuan Vuong | [github.com/mrz33-lab/london-tube-delay-prediction](https://github.com/mrz33-lab/london-tube-delay-prediction)

![Version](https://img.shields.io/badge/version-2.0.0-blue) ![Status](https://img.shields.io/badge/status-production--ready-brightgreen) ![Python](https://img.shields.io/badge/python-3.14.2-blue) ![Tests](https://img.shields.io/badge/tests-154%20passed%2C%201%20skipped-brightgreen) ![Updated](https://img.shields.io/badge/updated-April%202026-lightgrey)

---

Predicts delay severity (in minutes) for 11 London Underground lines using 33 days of real TfL operational data merged with concurrent OpenWeatherMap observations. The pipeline trains three models — a naive persistence baseline, Ridge regression, and LightGBM — on 40 engineered features with strict chronological train/test splits to prevent leakage. The system includes Optuna-style hyperparameter search, SHAP explainability, conformal calibration, a FastAPI backend, and an 8-tab Streamlit dashboard.

---

## 📊 Results

Evaluated on a held-out chronological test set (7,148 rows) from **run_20260413_123400** (seed 42).

| Model | Train MAE | Test MAE | Train RMSE | Test RMSE | Test R² | Test MAPE |
|---|---|---|---|---|---|---|
| Naive Baseline | 0.578 | 0.498 | 1.029 | 0.969 | −0.698 | 73.93% |
| Ridge Regression | 0.197 | 0.180 | 0.369 | 0.346 | 0.783 | 26.43% |
| **LightGBM** | **0.111** | **0.118** | **0.272** | **0.281** | **0.857** | **19.10%** |

All metrics are in minutes. Lower is better for MAE / RMSE / MAPE; higher is better for R².

**LightGBM achieves a 76.2% reduction in MAE over the naive persistence baseline.**

| Metric | Value |
|---|---|
| LightGBM Test MAE 95% CI | [0.104, 0.133] |
| LightGBM Test RMSE 95% CI | [0.254, 0.308] |
| Conformal calibration — median residual | 0.026 |
| Conformal calibration — q95 | 0.708 |

Bootstrap confidence intervals computed over the test set (seed 42). Hyperparameters tuned via RandomizedSearchCV with 5-fold TimeSeriesSplit (20 iterations).

### 🔍 SHAP Feature Importance

The four most predictive features are all lag/rolling historical delay statistics. `rolling_mean_delay_3` dominates global SHAP importance, contributing approximately **5× the impact of the next feature**. `hour_of_day` ranks 5th.

Weather features rank 1st in ablation analysis but not in SHAP — a documented analytical finding. This discrepancy reflects **correlation mediation through lag features**: weather conditions are partially captured by the recent delay history already, so their marginal SHAP contribution is lower than their independent ablation impact. This is not a pipeline error.

---

## 🗂️ Dataset

Real data collected 24/7 from two live APIs via a **systemd service on an Oracle Cloud Free Tier VM**:

| Property | Value |
|---|---|
| Sources | TfL Unified API + OpenWeatherMap |
| Collection window | 2026-03-10 00:21 → 2026-04-12 19:58 (~33 days) |
| Raw rows | 35,738 |
| Lines covered | 11 London Underground lines |
| Raw columns | 14 |
| Features after engineering | 40 |
| Train split | 28,590 rows (chronological) |
| Test split | 7,148 rows (chronological, no shuffling) |

Data lives at `data/tfl_merged.csv`. When this file is present the pipeline automatically uses real data. The collector (`data_collection.py`) polls both APIs every 15 minutes.

**Required environment variables:**
```
TFL_API_KEY=your_tfl_unified_api_key
WEATHER_API_KEY=your_openweathermap_api_key
```

**Crowding features** use static lookup tables (not real-time passenger counts). Ablation analysis shows zero measured marginal impact from the crowding feature group alone, which is consistent with the static approximation.

---

## ⚙️ Requirements

- Python 3.14.2
- LightGBM 4.6.0
- scikit-learn 1.8.0

```bash
git clone https://github.com/mrz33-lab/london-tube-delay-prediction
cd london-tube-delay-prediction
py -m venv .venv
.venv\Scripts\activate
py -m pip install -r requirements.txt
```

> **Note:** XGBoost is not available in the Python 3.14.2 environment. The pipeline detects this at runtime and skips XGBoost gracefully without error.

---

## 🚀 Usage

**1. Train all models**
```bash
py train.py
```
Runs the full pipeline: naive baseline → Ridge → LightGBM with RandomizedSearchCV tuning → bootstrap CIs → conformal calibration → per-line residual quantiles → classification evaluation. Outputs a timestamped artifact directory:
```
artifacts/run_YYYYMMDD_HHMMSS/
  ├── lgbm_model.pkl
  ├── metrics.json
  ├── residual_plots/
  ├── shap_values.pkl
  └── ...
```

**2. SHAP explainability**
```bash
py explain.py
```
Loads the latest artifact run and writes global SHAP summary plots, local force plots, and natural language feature importance summaries to the same artifact folder.

**3. Streamlit dashboard** (8 tabs)
```bash
py -m streamlit run app.py
```
Opens at `http://localhost:8501`

**4. FastAPI backend**
```bash
py -m uvicorn api:app --reload --port 8000
```
Interactive docs at `http://localhost:8000/docs`.  
Endpoints: `POST /predict`, `GET /health`, `GET /lines`

> **Prototype mode:** If `data/tfl_merged.csv` is absent, `data.py` falls back to a small synthetic dataset so the pipeline and dashboard remain runnable for demonstration purposes.

---

## 🧪 Tests

```bash
py -m pytest tests/ -v                          # 154 passed, 1 skipped
py -m pytest tests/ --cov=. --cov-report=html  # with HTML coverage report
```

| Category | Coverage |
|---|---|
| Schema validation | Column types, value ranges, null checks |
| Data leakage detection | Shift-before-roll ordering, no future data in features |
| Feature engineering | Per-line groupby correctness, lag/rolling outputs |
| API contract | Request/response shapes, error handling |

---

## 🗃️ Project Structure

```
├── config.py               # Central configuration (dataclasses)
├── data.py                 # Real / synthetic data loader
├── features.py             # 40-feature engineering with strict leakage
│                           #   protection (shift before rolling, per-line groupby)
├── train.py                # Full pipeline — naive + Ridge + LightGBM,
│                           #   TimeSeriesSplit CV, bootstrap CIs,
│                           #   conformal calibration, classification eval
├── explain.py              # SHAP global + local explainability,
│                           #   plots, natural language summaries
├── future_prediction.py    # Inference for unseen inputs
├── api.py                  # FastAPI service
├── app.py                  # 8-tab Streamlit dashboard
├── line_metadata.py        # Static TfL line metadata
├── data_collection.py      # Live TfL + weather collector (15-min polling)
├── tests/                  # 154 passing tests, 1 skipped
├── analysis/               # Ablation study, ARIMA baseline, learning curves
├── data/
│   └── tfl_merged.csv      # 33-day real TfL + OpenWeatherMap dataset
└── artifacts/              # Per-run outputs — models, metrics, plots (gitignored)
```

---

## 🔬 Methodology Summary

| Component | Detail |
|---|---|
| Data collection | TfL Unified API + OpenWeatherMap, Oracle Cloud Free Tier VM, systemd |
| Collection window | ~33 days (2026-03-10 to 2026-04-12), 15-min poll interval |
| Dataset | 35,738 rows, 11 lines, 14 raw columns → 40 engineered features |
| Target variable | Delay in minutes (regression) |
| Train / test split | Chronological — 28,590 train / 7,148 test (no shuffling) |
| Models | Naive persistence, Ridge regression, LightGBM |
| Hyperparameter tuning | RandomizedSearchCV, 5-fold TimeSeriesSplit, 20 iterations |
| Evaluation | MAE, RMSE, R², MAPE, bootstrap 95% CI, per-line residual quantiles |
| Conformal calibration | Median residual 0.026, q95 = 0.708 |
| Explainability | SHAP (global summary + local force plots) + ablation study |
| Environment | Python 3.14.2, LightGBM 4.6.0, scikit-learn 1.8.0 |
| Artifact run | run_20260413_123400, seed 42 |

---

## ⚠️ Known Limitations

- **33-day collection window**: data collection was constrained by infrastructure availability on Oracle Cloud Free Tier. Acknowledged in the dissertation critical reflection as a scope limitation affecting generalisation to seasonal variation.
- **Static crowding data**: crowding features are derived from fixed lookup tables rather than real-time passenger counts. Ablation analysis confirms zero measured marginal impact from the crowding feature group alone, consistent with this approximation.
- **SHAP vs. ablation discrepancy**: weather ranks 1st in ablation but does not dominate SHAP. This is a documented analytical finding — weather conditions are partially absorbed by lag features through correlation mediation, reducing their apparent marginal SHAP contribution. This is not a pipeline error.
- **XGBoost unavailable**: XGBoost is incompatible with Python 3.14.2 at the time of training. The pipeline detects the missing dependency at runtime and skips XGBoost gracefully.
