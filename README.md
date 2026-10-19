# Smart Bike Rental Demand Predictor

A beginner-friendly but professional end-to-end machine learning project. It predicts the **estimated number of bikes rented in an hour** from calendar, weather and season information. It includes data cleaning, preprocessing, model comparison, evaluation, a FastAPI backend, a Streamlit frontend, and a backend-integrated rule-based NLP entity extraction feature.

## Dataset used

**UCI Bike Sharing Dataset — hourly file `hour.csv`**

- Official page: https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset
- Historical Capital Bikeshare data from Washington, D.C. (2011–2012)
- 17,379 hourly records in the commonly distributed hourly file
- Target: `cnt`, the total number of bikes rented in that hour
- The project downloads the official archive on first training run and saves `hour.csv` to `data/raw/`.

If the automatic download fails, download the ZIP from the official page, extract `hour.csv`, and place it in `data/raw/hour.csv`.

## What the user enters

Season, year indicator, month, hour, holiday, weekday, working day, weather category, temperature (°C), feels-like temperature (°C), humidity (%) and wind speed (km/h). The application converts human-readable measurements into the normalized representation used in the original dataset.

> Note: The original dataset covers 2011–2012. Its year field is a historical indicator, not a current calendar-year forecast. Weather and time features are observational associations, not proof of causation.

## Algorithms compared

1. **Dummy Regressor** — baseline that predicts the training-set mean. More complex models should beat a simple baseline.
2. **Linear Regression** — a simple, interpretable starting point that models an additive relationship.
3. **Random Forest Regressor** — trains many decision trees on bootstrap samples and averages their predictions. Captures nonlinear relationships and feature interactions.
4. **Extra Trees Regressor** — trains randomized trees with additional split randomness; useful to compare against Random Forest.
5. **Hist Gradient Boosting Regressor** — sequentially adds trees to improve on previous errors.

The model with the **lowest validation RMSE** is selected. It is retrained using the training and validation periods together. The latest chronological 15% is kept aside as the final test period.

## ML workflow

`UCI CSV → inspect/clean → chronological split → preprocessing pipeline → train candidates → validation comparison → select best → refit → held-out test evaluation → save pipeline → FastAPI → Streamlit`

### Data cleaning and leakage prevention

- Parse the date column and sort records chronologically.
- Drop rows with invalid date or missing target; remove exact duplicate rows and negative target values.
- Inspect missingness and let the pipeline impute any feature gaps (median for numeric, most frequent for categorical).
- Scale numeric columns and one-hot encode categorical columns using a single fitted pipeline.
- Exclude `casual` and `registered`: their sum is `cnt`, so including them would reveal the answer (target leakage).
- Exclude `instant` (row identifier) and do not pass the raw date directly to the model.
- Split chronologically (70% train / 15% validation / 15% test) to make evaluation more realistic for time-ordered observations.

### Evaluation metrics

- **MAE:** average absolute error in number of bikes; lower is better.
- **RMSE:** gives larger errors more weight; lower is better.
- **R²:** compares the model with predicting the target mean; closer to 1 is generally better. It can be negative.

## Folder structure

```text
bike-rental-demand-predictor/
├── backend/main.py                 # FastAPI endpoints
├── frontend/app.py                 # Streamlit UI + optional text helper
├── src/data.py                     # Download/load/clean dataset
├── src/nlp_parser.py               # Rule-based NLP entity extraction
├── src/train.py                    # Split, train, compare, evaluate, save
├── data/raw/                       # hour.csv downloaded on first run
├── data/processed/                 # reserved for future processed exports
├── models/                         # saved pipeline created by training
├── reports/                        # comparison CSV, metrics JSON, test predictions
├── tests/                          # reserved for future tests
├── requirements.txt
└── README.md
```

## How to run on Windows

Use Python 3.10–3.12 for broad package compatibility. Open a terminal in the project root.

### 1. Create and activate virtual environment

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, use Command Prompt and run `.venv\Scripts\activate.bat` instead.

### 2. Train models (first run downloads the dataset)

```powershell
python -m src.train,uv run python -m src.train
```

This creates `models/bike_demand_model.joblib` and the reports in `reports/`. Read the validation comparison and final held-out test metrics; do not assume one model wins before running the experiment.

### 3. Start the backend

Open **Terminal 1** in the project root (with the venv activated):

```powershell
uvicorn backend.main:app --reload,
uv run uvicorn backend.main:app --reload
```

API docs: http://127.0.0.1:8000/docs
Health check: http://127.0.0.1:8000/health
NLP endpoint: `POST http://127.0.0.1:8000/parse-text`

### 4. Start the frontend

Open **Terminal 2**, activate the same venv, then run:

```powershell
streamlit run frontend/app.py,
uv run streamlit run frontend/app.py
```

Streamlit opens in your browser, normally at http://localhost:8501.

**Run order:** train once → start FastAPI → start Streamlit. Keep both terminals running while using the app.

## NLP: what is included?

The app includes a backend `/parse-text` endpoint powered by `src/nlp_parser.py`. Example input:

`Predict bike demand on a summer weekday at 8 AM, 25°C, with 60% humidity and clear weather.`

The parser uses regular expressions, phrase patterns and dictionaries to extract entities such as hour, temperature, humidity, season, month, weekday, working-day status, holiday status, weather, feels-like temperature, wind speed and dataset year when stated. The frontend fills the form with extracted values, displays what was recognized, and lets the user review/edit everything before prediction. If the month is omitted but a season is given, it chooses a representative month and explicitly warns the user.

**How to explain the NLP:** this is rule-based NLP / entity extraction, not a separately trained NLP model or an LLM. It is intentionally small, transparent and does not require a paid API or extra model download. A future extension could use spaCy NER or a trained intent/entity model, but the core project should first demonstrate reliable tabular ML, leakage prevention, preprocessing, validation and evaluation.

## Important limitations

- Data is historical and comes from one city in 2011–2012; results may not transfer to another city or today's conditions.
- The target is the observed number of rentals, not unmet demand or the number of bikes that should be stocked.
- Predictions are estimates, not guaranteed counts.
- The app converts Celsius, humidity and wind speed into the original dataset's normalized features. The source data's normalized temperature is mapped using `(temperature_C + 8) / 47`, feels-like temperature using `(feels_like_C + 16) / 66`, humidity using `/100`, and wind speed in m/s using `/67`.
- For an academic report, cite the UCI dataset and explain the time-based split and leakage checks.

## Dataset citation

Fanaee-T, H. & Gama, J. (2013). *Event labeling combining ensemble detectors and background knowledge*. Progress in Artificial Intelligence. Dataset: UCI Machine Learning Repository, Bike Sharing Dataset, https://doi.org/10.24432/C5W894.
