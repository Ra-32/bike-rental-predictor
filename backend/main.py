from pathlib import Path
import sys
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.nlp_parser import parse_bike_request

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "bike_demand_model.joblib"
sys.path.insert(0, str(ROOT))
app = FastAPI(title="Smart Bike Demand API", version="1.0.0", description="Predict hourly bike rentals using a trained regression model.")

class TextParseInput(BaseModel):
    text: str = Field(min_length=1, max_length=1000, description="Plain-English bike demand conditions")

class PredictionInput(BaseModel):
    season: int = Field(ge=1, le=4, description="1=spring, 2=summer, 3=fall, 4=winter")
    year: int = Field(ge=0, le=1, description="0=2011, 1=2012 in training dataset")
    month: int = Field(ge=1, le=12)
    hour: int = Field(ge=0, le=23)
    holiday: int = Field(ge=0, le=1)
    weekday: int = Field(ge=0, le=6, description="0=Sunday, 6=Saturday in this dataset")
    workingday: int = Field(ge=0, le=1)
    weather: int = Field(ge=1, le=4, description="1=clear, 2=mist/cloudy, 3=light rain/snow, 4=heavy rain/snow")
    temperature_c: float = Field(ge=-8, le=39)
    feels_like_c: float = Field(ge=-16, le=50)
    humidity_percent: float = Field(ge=0, le=100)
    wind_speed_kmh: float = Field(ge=0, le=241.2)

@app.get("/")
def root():
    return {"message": "Smart Bike Demand API", "docs": "/docs", "health": "/health"}

@app.post("/parse-text")
def parse_text(payload: TextParseInput):
    """Extract likely input fields from plain English using rule-based NLP."""
    result = parse_bike_request(payload.text)
    if not result["fields"]:
        result["message"] = "No supported conditions were recognized. Try including an hour, temperature, humidity, season, weekday, or weather."
    return result

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL_PATH.exists()}

@app.post("/predict")
def predict(payload: PredictionInput):
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=503, detail="Model not trained yet. Run: python -m src.train")
    bundle = joblib.load(MODEL_PATH)
    # Convert human-friendly units back to the normalized units used by UCI.
    row = {
        "temp": (payload.temperature_c + 8) / 47,
        "atemp": (payload.feels_like_c + 16) / 66,
        "hum": payload.humidity_percent / 100,
        "windspeed": (payload.wind_speed_kmh / 3.6) / 67,
        "season": payload.season, "yr": payload.year, "mnth": payload.month,
        "hr": payload.hour, "holiday": payload.holiday, "weekday": payload.weekday,
        "workingday": payload.workingday, "weathersit": payload.weather,
    }
    try:
        prediction = float(bundle["pipeline"].predict(pd.DataFrame([row]))[0])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc
    return {"predicted_rentals": max(0, int(round(prediction))), "model": bundle["model_name"],
            "test_metrics": bundle["test_metrics"], "note": "Estimate based on historical 2011-2012 Washington, D.C. data; not a live forecast."}
