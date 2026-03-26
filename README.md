# London Tube Delay Prediction

Predicts delay severity (in minutes) for 11 London Underground lines using weather, crowding, network, and time-of-day data. Trained with LightGBM on 37 engineered features, served via FastAPI, with a Streamlit dashboard and SHAP explainability.

---

## Requirements

- Python 3.8+

```bash
git clone https://github.com/mrz33-lab/london-tube-delay-prediction
cd london-tube-delay-prediction
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Running

**Train a model**
```bash
python train.py
```
Outputs artifacts to `artifacts/run_YYYYMMDD_HHMMSS/` — metrics, plots, saved model.

**Dashboard**
```bash
streamlit run app.py
```
Opens at `http://localhost:8501`

**API**
```bash
uvicorn api:app --reload
```
Docs at `http://localhost:8000/docs`. Endpoints: `POST /predict`, `GET /health`, `GET /lines`.

**SHAP explanations**
```bash
python explain.py
```
Loads the latest run and writes SHAP plots + natural language summaries to its artifact folder.

---

## Real TfL Data (optional)

By default the pipeline trains on synthetic data. To use real data:

1. Copy `.env.example` to `.env` and add your OpenWeatherMap API key
2. Run the setup wizard: `python scripts/setup_data_collection.py`
3. Start collecting: `python data_collection.py` (polls every 15 minutes)

Once `data/tfl_merged.csv` exists the pipeline switches to real mode automatically.

---

## Tests

```bash
pytest tests/ -v
pytest tests/ --cov=. --cov-report=html  # with coverage
```

---

## Project Structure

```
├── config.py               # Central config
├── data.py                 # Data loading / synthetic generation
├── features.py             # Feature engineering (37 features)
├── train.py                # Training pipeline
├── explain.py              # SHAP pipeline
├── future_prediction.py    # Inference
├── api.py                  # FastAPI service
├── app.py                  # Streamlit dashboard
├── line_metadata.py        # Static TfL line metadata
├── data_collection.py      # Real-time TfL + weather collector
├── tests/                  # Test suite
├── analysis/               # Ablation, ARIMA baseline, learning curves
└── artifacts/              # Model runs (gitignored)
```
