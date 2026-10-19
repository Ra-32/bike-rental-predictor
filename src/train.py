"""Train, compare, evaluate and save bike-demand regression pipelines."""
from pathlib import Path
import json
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import load_data

NUMERIC = ["temp", "atemp", "hum", "windspeed"]
CATEGORICAL = ["season", "yr", "mnth", "hr", "holiday", "weekday", "workingday", "weathersit"]
FEATURES = NUMERIC + CATEGORICAL
TARGET = "cnt"


def build_pipeline(model):
    preprocess = ColumnTransformer([
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), NUMERIC),
        ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), CATEGORICAL),
    ])
    return Pipeline([("preprocess", preprocess), ("model", model)])


def metrics(y_true, y_pred):
    return {
        "MAE": round(float(mean_absolute_error(y_true, y_pred)), 3),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 3),
        "R2": round(float(r2_score(y_true, y_pred)), 4),
    }


def main():
    (ROOT / "models").mkdir(exist_ok=True)
    (ROOT / "reports").mkdir(exist_ok=True)
    df = load_data()
    print("\nDATA QUALITY CHECK")
    print(f"Rows after cleaning: {len(df):,}")
    print(f"Duplicate rows remaining: {df.duplicated().sum()}")
    print(f"Missing cells in selected features: {int(df[FEATURES].isna().sum().sum())}")
    print("Target: cnt = total hourly rentals")
    # Exclude instant, dteday, casual and registered. casual + registered = cnt,
    # so including either would leak the answer into the features.
    X = df[FEATURES].copy()
    y = df[TARGET].astype(float).copy()
    n = len(df)
    train_end, val_end = int(n * .70), int(n * .85)
    X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
    X_val, y_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
    X_test, y_test = X.iloc[val_end:], y.iloc[val_end:]
    print(f"Chronological split: train={len(X_train):,}, validation={len(X_val):,}, test={len(X_test):,}")

    candidates = {
        "Dummy baseline": DummyRegressor(strategy="mean"),
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=250, min_samples_leaf=2, random_state=42, n_jobs=-1),
        "Extra Trees": ExtraTreesRegressor(n_estimators=250, min_samples_leaf=2, random_state=42, n_jobs=-1),
        "Hist Gradient Boosting": HistGradientBoostingRegressor(max_iter=180, learning_rate=0.08, l2_regularization=1.0, random_state=42),
    }
    results = []
    fitted = {}
    for name, estimator in candidates.items():
        pipe = build_pipeline(estimator)
        pipe.fit(X_train, y_train)
        pred = np.maximum(0, pipe.predict(X_val))
        row = {"Model": name, **metrics(y_val, pred)}
        results.append(row)
        fitted[name] = pipe
        print(f"{name:24s} validation: {row}")
    results_df = pd.DataFrame(results).sort_values("RMSE")
    results_df.to_csv(ROOT / "reports" / "model_comparison.csv", index=False)
    best_name = str(results_df.iloc[0]["Model"])
    # Refit the selected algorithm on train + validation; test remains unseen until final evaluation.
    X_trainval = pd.concat([X_train, X_val], axis=0)
    y_trainval = pd.concat([y_train, y_val], axis=0)
    final_pipe = build_pipeline(candidates[best_name])
    final_pipe.fit(X_trainval, y_trainval)
    test_pred = np.maximum(0, final_pipe.predict(X_test))
    test_metrics = metrics(y_test, test_pred)
    bundle = {"pipeline": final_pipe, "features": FEATURES, "numeric_features": NUMERIC,
              "categorical_features": CATEGORICAL, "target": TARGET, "model_name": best_name,
              "test_metrics": test_metrics, "dataset": "UCI Bike Sharing Dataset (hour.csv)"}
    joblib.dump(bundle, ROOT / "models" / "bike_demand_model.joblib")
    (ROOT / "reports" / "final_metrics.json").write_text(json.dumps({"selected_model": best_name, "test_metrics": test_metrics}, indent=2))
    # Save test predictions for dashboard visualizations; these are genuinely held-out predictions.
    pd.DataFrame({"actual": y_test.to_numpy(), "predicted": test_pred}).to_csv(ROOT / "reports" / "test_predictions.csv", index=False)
    print("\nMODEL SELECTION (lowest validation RMSE)")
    print(results_df.to_string(index=False))
    print(f"\nSelected model: {best_name}")
    print(f"Held-out test metrics: {test_metrics}")
    print("Saved: models/bike_demand_model.joblib")
    print("Saved reports/model_comparison.csv, final_metrics.json, test_predictions.csv")

if __name__ == "__main__":
    main()
