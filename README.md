# London Underground Delay Prediction System

Machine learning pipeline for predicting London Underground delay severity using weather, crowding, network, and temporal data.

**Course**: COMP1682 Final Year Project — University of Greenwich

---

## Table of Contents

- [Overview](#overview)
- [Requirements Specification](#requirements-specification)
- [Comparison of Similar Systems](#comparison-of-similar-systems)
- [Data Schema](#data-schema)
- [Feature Engineering](#feature-engineering)
- [Modeling Strategy](#modeling-strategy)
- [Explainability](#explainability)
- [Real Data Collection](#real-data-collection)
- [Development Methodology](#development-methodology)
- [Development Timeline](#development-timeline)
- [Installation](#installation)
- [Usage](#usage)
- [Testing Strategy](#testing-strategy)
- [Project Structure](#project-structure)
- [Design Decisions](#design-decisions)
- [Limitations](#limitations)
- [Legal, Social, Ethical and Professional Issues](#legal-social-ethical-and-professional-issues)
- [References](#references)
- [Critical Analysis](#critical-analysis)

---

## Overview

End-to-end ML pipeline that predicts delay severity (minutes) for 11 London Underground lines. Can run on synthetic data or real TfL data.

- Strict time-aware validation — no data leakage
- SHAP explainability
- FastAPI REST service for predictions
- Streamlit dashboard for visualisation

---

## Requirements Specification

Requirements are classified using the MoSCoW method. Functional requirements (FR) describe system behaviour; non-functional requirements (NFR) constrain quality attributes.

### Functional Requirements

| ID | Requirement | Priority | Status |
|----|-------------|----------|--------|
| FR-01 | The system shall predict delay severity (in minutes) for each of the 11 London Underground lines | Must | Done |
| FR-02 | The system shall engineer features from temporal, weather, crowding, and network-effect data with no data leakage | Must | Done |
| FR-03 | The system shall compare at least three model types (naive baseline, linear, gradient boosting) and select the best performer | Must | Done |
| FR-04 | The system shall expose predictions via a RESTful API with JSON request/response | Must | Done |
| FR-05 | The system shall provide an interactive dashboard for visualising predictions and model diagnostics | Must | Done |
| FR-06 | The system shall generate SHAP explanations for global and local model transparency | Must | Done |
| FR-07 | The system shall collect real TfL and weather data at 15-minute intervals | Should | Done |
| FR-08 | The system shall provide batch prediction and 24-hour forecast endpoints | Should | Done |
| FR-09 | The system shall generate confidence intervals for each prediction using bootstrap resampling | Should | Done |
| FR-10 | The system shall support Optuna-based hyperparameter optimisation as an alternative to RandomizedSearchCV | Could | Not done |
| FR-11 | The system shall produce per-status confusion matrices alongside regression metrics | Could | Not done |
| FR-12 | The system shall support real-time streaming predictions via WebSocket | Will not | -- |

### Non-Functional Requirements

| ID | Requirement | Priority | Status |
|----|-------------|----------|--------|
| NFR-01 | The training pipeline shall complete within 10 minutes on commodity hardware | Must | Met (~3 min) |
| NFR-02 | The API shall respond to single predictions within 200ms | Must | Met (~50ms) |
| NFR-03 | All lag/rolling features shall pass automated leakage verification on every run | Must | Met (61 tests) |
| NFR-04 | The system shall be containerisable via Docker | Should | Met (Dockerfile) |
| NFR-05 | The system shall use type hints throughout and pass static analysis | Should | Met (py.typed) |
| NFR-06 | CI shall run the full test suite on every push and pull request | Should | Met (GitHub Actions) |
| NFR-07 | All configuration shall be centralised in a single importable module | Should | Met (config.py) |
| NFR-08 | The system shall achieve a test MAE below the naive baseline | Must | Met (2.01 vs 3.62) |

---

## Comparison of Similar Systems

| System | Approach | Strengths | Limitations vs This Project |
|--------|----------|-----------|-----------------------------|
| **TfL Status API** | Rule-based; reports current status from signal systems | Official source, real-time, 100% coverage | Reactive only -- no prediction; binary Good/Delayed with no numeric severity |
| **Citymapper** | Aggregates TfL feeds + user reports; shows disruptions on a map | Excellent UI, crowd-sourced incident reports | No ML prediction; relies on same TfL status with no forward-looking estimates |
| **Google Maps Transit** | Real-time transit departure boards using GTFS-RT feeds | Global coverage, integrated routing | Does not predict delays -- only shows scheduled vs live departure times |
| **Silva et al. (2022)** | Random Forest on Sao Paulo bus GPS data (see References) | Demonstrated RF viability for urban transit delays | Different city dynamics; no weather features; no explainability |
| **Wen et al. (2019)** | GAN-based anomaly detection on metro ridership (see References) | Novel deep learning approach for outlier detection | Detects anomalies, not severity; requires large labelled anomaly datasets |
| **This project** | LightGBM with 37 engineered features, SHAP explainability, FastAPI + Streamlit | End-to-end pipeline from data collection to interactive dashboard; per-line forecasts with confidence intervals; full leakage protection; SHAP transparency | Limited to synthetic training data; delay estimates inferred from status categories |

The main things that set this apart from existing tools are: (a) it's a full pipeline from data collection through to deployment, (b) leakage protection is verified by automated tests, and (c) SHAP explanations are built in rather than bolted on.

---

## Data Schema

Each row is a line-level observation at a specific timestamp.

| Column | Type | Description | Range/Values |
|--------|------|-------------|--------------|
| `timestamp` | datetime | Observation time | — |
| `line` | categorical | Tube line | 11 lines |
| `status` | categorical | Service status | Good Service, Minor Delays, Severe Delays |
| `delay_minutes` | float | **TARGET** | ≥ 0 |
| `temp_c` | float | Temperature (°C) | −5 to 35 |
| `precipitation_mm` | float | Precipitation (mm) | ≥ 0 |
| `humidity` | float | Relative humidity (%) | 0–100 |
| `crowding_index` | float | Crowding level | 0–1 |
| `is_weekend` | int | Weekend indicator | 0 or 1 |
| `hour` | int | Hour of day | 0–23 |
| `day_of_week` | int | Day of week (Monday=0) | 0–6 |
| `month` | int | Month | 1–12 |
| `peak_time` | int | Peak hours 07–10, 16–19 | 0 or 1 |
| `is_holiday` | int | UK bank holiday | 0 or 1 |

**Data modes:**
- **Real mode**: loads `data/tfl_merged.csv` when present
- **Prototype mode**: generates synthetic data matching the schema

---

## Feature Engineering

All features are computed with explicit leakage protection. The pipeline produces **37 features** across eight groups (expanded from 21 in v1.0).

### 1. Lag Features
| Feature | Description |
|---------|-------------|
| `lag_delay_1` | Delay 1 hour ago |
| `lag_delay_3` | Delay 3 hours ago |

### 2. Rolling Statistics
| Feature | Description |
|---------|-------------|
| `rolling_mean_delay_3` | Mean delay over past 3 hours |
| `rolling_mean_delay_12` | Mean delay over past 12 hours |
| `rolling_std_delay_12` | Std deviation over past 12 hours |

### 3. Disruption Metrics
| Feature | Description |
|---------|-------------|
| `recent_disruption_rate` | Fraction of recent non-Good Service observations |

### 4. Weather Signals
| Feature | Description |
|---------|-------------|
| `temp_delta_1h` | Temperature change from 1 hour ago |
| `precipitation_delta_1h` | Precipitation change from 1 hour ago |
| `crowding_x_peak` | Crowding index × peak-time indicator |
| `precipitation_x_temp` | Weather interaction effect |

### 5. Network Effect Features *(added March 2026)*
Leave-one-out — computed from other lines at the same timestamp, so valid at prediction time.

| Feature | Description |
|---------|-------------|
| `network_avg_delay` | Mean delay of all other lines |
| `network_delay_volatility` | Std of delay across other lines |
| `lines_disrupted_ratio` | Fraction of other lines disrupted |
| `is_network_wide_disruption` | 1 if ≥ 50% of other lines are disrupted |

### 6. Special Event / Temporal Features *(added March 2026)*
Cyclical encoding prevents the model treating midnight and 23:00 as distant.

| Feature | Description |
|---------|-------------|
| `hour_sin` | sin(2π × hour / 24) |
| `hour_cos` | cos(2π × hour / 24) |
| `is_late_night` | 1 if hour ≥ 22 |
| `is_early_morning` | 1 if hour < 6 |

### 7. Station Topology Features *(added March 2026)*
Static infrastructure metadata from `line_metadata.py` — no temporal leakage.

| Feature | Description |
|---------|-------------|
| `line_length_km` | Total route length |
| `n_stations` | Number of stops |
| `n_interchange_stations` | Stations with cross-line transfers |
| `is_deep_tube` | 1 = deep-bore, 0 = sub-surface |
| `zone_coverage` | Fare zones served |

### 8. Train Frequency Features *(added March 2026)*
Supply-side signals — low-frequency lines are more vulnerable to cascade failures.

| Feature | Description |
|---------|-------------|
| `trains_per_hour` | Scheduled TPH for line and time of day |
| `service_headway_min` | Minutes between trains (60 / TPH) |
| `capacity_pressure` | crowding_index × TPH / max_TPH |

### Leakage Protection

- All lag and rolling features computed **per line** via `groupby`, using `shift()` before any rolling window
- Network features use **leave-one-out** aggregation
- Topology and frequency features are static lookups

```python
# Lag: shift(1) ensures only past data is used
df['lag_delay_1'] = df.groupby(group_col)[target_col].shift(1)

# Rolling: shift(1) before rolling() excludes the current value
df['rolling_mean_delay_3'] = df.groupby(group_col)[target_col].transform(
    lambda x: x.shift(1).rolling(window=3, min_periods=1).mean()
)

# Network: leave-one-out mean (other lines only)
loo_sum = df['net_sum'] - df[target_col]
df['network_avg_delay'] = loo_sum / (df['net_count'] - 1).clip(lower=1)
```

---

## Modeling Strategy

**Task**: regression on `delay_minutes`.

### Models Compared

| Model | Description |
|-------|-------------|
| Naive baseline | Predicts last observed delay per line |
| Ridge regression | Linear model with L2 regularisation, scaling, one-hot encoding |
| LightGBM / XGBoost / RandomForest | Gradient boosting (priority order); best performer selected automatically |

### Validation

- Chronological 80/20 train/test split — no shuffling
- `TimeSeriesSplit` (5 folds) for cross-validation
- `RandomizedSearchCV` (20 iterations) for hyperparameter tuning
- Primary metric: MAE; also reports RMSE, R², and 95% bootstrap CIs

---

## Explainability

SHAP (SHapley Additive exPlanations) is used for model transparency.

- **Global**: beeswarm plot, bar plot, feature importance CSV
- **Local**: waterfall plots and natural language summaries per prediction

All outputs saved to `artifacts/<run_id>/`.

---

## Real Data Collection

The pipeline trains on synthetic data by default. Real TfL data is used for dissertation validation.

### Data Sources

| Source | Data | Auth |
|--------|------|------|
| TfL Unified API | Live line status | Free; key optional |
| OpenWeatherMap | Current London weather | Free; key required (≤ 1,000 calls/day) |

### Setup

```bash
# 1. Run the interactive setup wizard (tests both APIs, writes .env)
python scripts/setup_data_collection.py

# 2. Test a single snapshot (11 records, one per line)
python data_collection.py --once

# 3. Start continuous collection (every 15 minutes)
python data_collection.py

# 4. Monitor progress
python scripts/check_collection_progress.py
```

Target: 14,784 records (2 weeks). The pipeline auto-detects `data/tfl_merged.csv` and switches to real mode.

### Troubleshooting

| Symptom | Fix |
|---------|-----|
| `WEATHER_API_KEY is not set` | Re-run the setup wizard |
| `HTTP 401` from weather API | Wait 10 min for key activation, then retry |
| Large collection gaps | Restart `data_collection.py` |

API keys are stored in `.env` (gitignored). See `.env.example` for the template.

---

## Development Methodology

This project followed an **iterative incremental** approach loosely based on Agile, adapted for solo development.

### Iteration Structure

| Iteration | Focus | Key Deliverables |
|-----------|-------|------------------|
| 1 (Weeks 1-3) | Foundation | Project setup, data schema design, synthetic data generator, config system |
| 2 (Weeks 4-6) | Core ML pipeline | Feature engineering (21 features), naive/Ridge/LightGBM training, evaluation |
| 3 (Weeks 7-9) | Explainability and API | SHAP pipeline, FastAPI service, Streamlit dashboard |
| 4 (Weeks 10-12) | Feature expansion and real data | 16 new features (37 total), TfL data collection, leakage verification |
| 5 (Weeks 13-14) | Analysis and documentation | Ablation study, ARIMA baseline, learning curves, critical analysis, CI/CD |

### Evidence of Iterative Development

- **Git history** tracks incremental commits across all modules
- **Two artifact runs** evidence separate training iterations: `run_20260210_153030` (prototype) and `run_20260310_164049` (real data)
- **Feature count evolution**: 21 features (v1.0) to 37 features (v1.2) documented in `features.py` and `analysis/improvement_history.py`
- **Analysis scripts** in `analysis/` were added iteratively as each model evaluation revealed questions

---

## Development Timeline

```
Week  1 ----..........  Project setup, config.py, data schema
Week  2 ----..........  Synthetic data generator, schema validation
Week  3 ------........  Feature engineering v1 (21 features)
Week  4 --------......  Naive + Ridge baselines, TimeSeriesSplit
Week  5 ----------....  LightGBM integration, hyperparameter tuning
Week  6 ------------..  SHAP explainability pipeline
Week  7 --------------  FastAPI REST service
Week  8 --------------  Streamlit dashboard (TfL branding)
Week  9 ------------..  TfL + weather data collection pipeline
Week 10 ----------....  Network effect features, topology features
Week 11 --------......  Train frequency features, feature expansion to 37
Week 12 ------........  Analysis scripts (ablation, ARIMA, learning curves)
Week 13 ----..........  CI/CD (GitHub Actions), Dockerfile, testing
Week 14 ----..........  Critical analysis, documentation, final polish
```

Key milestones were mostly on schedule. The main hold-up was week 9 — real data collection took longer than expected because of TfL API rate limiting, so only 44 real observations were collected vs the target of 14,784.

---

## Installation

**Requirements**: Python 3.8+

```bash
# Clone the repository
git clone <repo-url>
cd london-tube-delay-prediction

# Create virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### Train

```bash
python train.py
```

Loads or generates data, engineers features, trains all models, evaluates, and saves artifacts to `artifacts/run_YYYYMMDD_HHMMSS/`.

### Explain

```bash
python explain.py
```

Loads the latest run, computes SHAP values, and writes plots and natural language summaries.

### Dashboard

```bash
streamlit run app.py
# Open http://localhost:8501
```

### REST API

```bash
uvicorn api:app --reload
# Open http://localhost:8000/docs
```

Endpoints: `POST /predict`, `GET /health`, `GET /lines`.

### Tests

```bash
# All tests
pytest tests/ -v

# Individual files
pytest tests/test_schema.py -v
pytest tests/test_leakage.py -v
pytest tests/test_features.py -v
pytest tests/test_api.py -v
pytest tests/test_future_prediction.py -v

# With coverage
pytest tests/ --cov=. --cov-report=html
```

### Testing Strategy

The test suite contains **61 tests** across 6 test files, organised into three tiers:

| Tier | Files | What is tested |
|------|-------|----------------|
| Schema validation | `test_schema.py` | Required columns, data types, value ranges, status categories |
| Leakage detection | `test_leakage.py` | Lag features are NaN at series start, no future information in rolling windows |
| Feature engineering | `test_features.py` | Correct feature count (37), proper groupby behaviour, edge cases |
| API contracts | `test_api.py` | HTTP status codes, request validation, prediction response schema |
| Inference pipeline | `test_future_prediction.py` | End-to-end predict to confidence interval to status label |
| Explainability | `test_explain.py` | SHAP value computation, plot generation, text explanations |

**Why leakage tests specifically?** Data leakage is probably the most dangerous ML bug — the model looks great in training but fails silently in production. The leakage tests check that:
1. Every lag feature's first value per line is NaN (so `shift()` was actually applied)
2. Rolling features exclude the current row
3. Network features use leave-one-out aggregation

**Coverage note:** The existing `htmlcov/` report shows 1% line coverage because `coverage.py` was run without `--source` targeting -- it measured coverage across all project files including `app.py` (529 statements) and `data_collection.py` (296 statements) which are not unit-testable. Running `pytest --cov=features --cov=data --cov=api --cov=utils` produces a more representative figure.

---

## Project Structure

```
london-tube-delay-prediction/
|
|-- config.py                # Central configuration (dataclasses)
|-- line_metadata.py         # Static TfL line metadata
|-- data.py                  # Data loading and synthetic generation
|-- data_collection.py       # Real-time TfL + weather collector
|-- features.py              # Feature engineering (37 features)
|-- train.py                 # Model training pipeline
|-- explain.py               # SHAP explainability pipeline
|-- future_prediction.py     # Inference-time predictions
|-- api.py                   # FastAPI REST service
|-- app.py                   # Streamlit dashboard
|-- utils.py                 # Shared helpers
|-- requirements.txt
|-- Dockerfile               # Production container
|-- py.typed                 # PEP 561 type hint marker
|-- CRITICAL_ANALYSIS.md     # In-depth performance analysis
|-- .env.example             # API key template
|
|-- credentials/
|   +-- api_config.py        # APIKeys dataclass, env/dotenv loader
|
|-- analysis/
|   |-- ablation_study.py
|   |-- arima_baseline.py
|   |-- learning_curves.py
|   |-- per_line_performance.py
|   |-- confidence_interval_calibration.py
|   |-- improvement_history.py
|   +-- outputs/              # Generated plots
|
|-- scripts/
|   |-- setup_data_collection.py
|   |-- check_collection_progress.py
|   +-- deprecated/           # Archived fix scripts
|
|-- tests/
|   |-- conftest.py
|   |-- test_schema.py
|   |-- test_leakage.py
|   |-- test_features.py
|   |-- test_api.py
|   |-- test_explain.py
|   +-- test_future_prediction.py
|
|-- .github/workflows/
|   +-- tests.yml             # CI pipeline
|
|-- data/
|   +-- tfl_merged.csv       # Real data (gitignored)
|
+-- artifacts/
    +-- run_YYYYMMDD_HHMMSS/
        |-- config.yaml
        |-- all_metrics.json
        |-- model_comparison.csv
        |-- test_predictions.csv
        |-- feature_importance.csv
        |-- X_train.parquet / X_test.parquet
        +-- *.png
```

---

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| Dataclasses for config | Type safety, IDE support, easy extension |
| `ensure_dirs()` separate from `__post_init__` | Prevents directory side-effects on import |
| FastAPI lifespan context manager | Model loaded once at startup, not per request |
| Leave-one-out network features | Valid at inference time without target leakage |
| `TimeSeriesSplit` validation | Respects temporal ordering for realistic estimates |
| Multiple baselines | Quantifies the value added by each level of complexity |

---

## Limitations

- **Synthetic data**: prototype mode may not capture all real-world patterns
- **Delay estimation**: TfL free API does not return exact delay durations; values are modelled from status categories
- **Static topology**: `line_metadata.py` reflects March 2026 network; needs updating if infrastructure changes
- **Temporal resolution**: assumes regular 15-minute intervals; irregular gaps require preprocessing
- **No real-time serving**: batch prediction only; streaming would require additional infrastructure

---

## Legal, Social, Ethical and Professional Issues

### Legal

- **GDPR**: No personal data is processed — everything is aggregated line-level operational data. No passenger info is collected.
- **API Terms**: TfL data is used under the Open Government Licence v3.0. OpenWeatherMap under the free tier (≤1,000 calls/day).
- **IP**: All libraries are open-source (MIT/BSD/Apache 2.0).

### Social

- **Accessibility**: Dashboard uses TfL branding with high contrast. Delay severity uses both colour and text labels.
- **Equity**: All 11 lines are treated equally, though lines serving different demographics may have different delay patterns — this isn't accounted for.
- **Public benefit**: Better delay predictions help commuters plan ahead.

### Ethical

- **Transparency**: SHAP explanations are generated for every prediction so the model's reasoning is auditable.
- **Human oversight**: This is a decision-support tool, not an autonomous system. Predictions include confidence intervals.
- **Bias**: Synthetic training data may not reflect real correlations. Should be retrained on real data before any serious deployment.

### Professional

- **BCS Code of Conduct**: Code is documented, tested, and version-controlled in line with BCS principles.
- **Reproducibility**: Each run saves its config, random seeds, and evaluation artifacts.
- **Code quality**: Type hints, docstrings, CI testing.

---

## References

1. Lundberg, S.M. and Lee, S.-I. (2017) 'A Unified Approach to Interpreting Model Predictions', *Advances in Neural Information Processing Systems*, 30.
2. Chen, T. and Guestrin, C. (2016) 'XGBoost: A Scalable Tree Boosting System', *Proceedings of the 22nd ACM SIGKDD*, pp. 785-794.
3. Ke, G. et al. (2017) 'LightGBM: A Highly Efficient Gradient Boosting Decision Tree', *Advances in Neural Information Processing Systems*, 30.
4. Bergmeir, C. and Benitez, J.M. (2012) 'On the use of cross-validation for time series predictor evaluation', *Information Sciences*, 191, pp. 192-213.
5. Silva, C., Masiero, B. and Arruda, J. (2022) 'Predicting Bus Delays with Machine Learning', *Transportation Research Record*, 2676(5), pp. 87-98.
6. Wen, T., Keyes, R. and Gkiotsalitis, K. (2019) 'Anomaly Detection in Metro Ridership Data Using GANs', *IEEE Transactions on Intelligent Transportation Systems*, 21(10), pp. 4346-4357.
7. Transport for London (2024) *Unified API Documentation*. Available at: https://api-portal.tfl.gov.uk/docs (Accessed: March 2026).
8. Transport for London (2023) *Travel in London Report 15*. Available at: https://tfl.gov.uk/corporate/publications-and-reports/travel-in-london-reports
9. Cerqueira, V., Torgo, L. and Mozetic, I. (2020) 'Evaluating time series forecasting models: an empirical study on performance estimation methods', *Machine Learning*, 109, pp. 1997-2028.
10. Breiman, L. (2001) 'Random Forests', *Machine Learning*, 45(1), pp. 5-32.
11. Hastie, T., Tibshirani, R. and Friedman, J. (2009) *The Elements of Statistical Learning*. 2nd edn. New York: Springer.
12. Akiba, T. et al. (2019) 'Optuna: A Next-generation Hyperparameter Optimization Framework', *Proceedings of the 25th ACM SIGKDD*, pp. 2623-2631.
13. Ribeiro, M.T., Singh, S. and Guestrin, C. (2016) 'Why Should I Trust You?: Explaining the Predictions of Any Classifier', *Proceedings of the 22nd ACM SIGKDD*, pp. 1135-1144.
14. Molnar, C. (2022) *Interpretable Machine Learning: A Guide for Making Black Box Models Explainable*. 2nd edn. Available at: https://christophm.github.io/interpretable-ml-book/
15. Noursalehi, P., Koutsopoulos, H.N. and Zhao, J. (2018) 'Real-Time Predictive Analytics for Improving Public Transportation Systems Resilience', *Transportation Research Record*, 2672(8), pp. 26-37.
16. Pedregosa, F. et al. (2011) 'Scikit-learn: Machine Learning in Python', *Journal of Machine Learning Research*, 12, pp. 2825-2830.
17. Hyndman, R.J. and Athanasopoulos, G. (2021) *Forecasting: Principles and Practice*. 3rd edn. Melbourne: OTexts.
18. Goodfellow, I., Bengio, Y. and Courville, A. (2016) *Deep Learning*. Cambridge: MIT Press.
19. BCS (2022) *Code of Conduct*. Available at: https://www.bcs.org/membership-and-registrations/become-a-member/bcs-code-of-conduct/
20. UK Government (2018) *Data Protection Act 2018*. Available at: https://www.legislation.gov.uk/ukpga/2018/12/contents/enacted

---

## Critical Analysis

A full critical analysis of the model's performance, limitations, and design decisions is in
[`CRITICAL_ANALYSIS.md`](CRITICAL_ANALYSIS.md). It covers feature importance (ablation study),
per-line error breakdown, regression vs. ordinal classification justification, risky assumptions,
ethical considerations, and future work.

The supporting analysis scripts in [`analysis/`](analysis/) can be run from the project root:

```bash
python analysis/per_line_performance.py
python analysis/ablation_study.py
python analysis/arima_baseline.py
python analysis/learning_curves.py
python analysis/confidence_interval_calibration.py
python analysis/improvement_history.py
```

---

*Last updated: March 2026*
