from pathlib import Path
import json
import sys
import requests
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
API_URL = "http://127.0.0.1:8000"
st.set_page_config(page_title="Smart Bike Demand", page_icon="🚲", layout="wide")
st.title("🚲 Smart Bike Rental Demand Predictor")
st.caption("End-to-end machine learning • Regression • Rule-based NLP • FastAPI + Streamlit")

DEFAULTS = {
    "season_label": "Summer", "month": 6, "hour": 8, "weekday_label": "Monday",
    "weather_label": "Clear / partly cloudy", "temperature": 20.0, "feels_like": 21.0,
    "humidity": 55.0, "wind": 10.0, "holiday_label": "No", "workingday_label": "Yes",
    "year_label": "2012 (1)", "nlp_text": "Predict bike demand on a summer weekday at 8 AM, 25°C, with 60% humidity and clear weather.",
}
for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value

with st.sidebar:
    st.header("About this project")
    st.write("Predict hourly bike rentals from calendar and weather conditions.")
    st.info("Dataset: UCI Bike Sharing Dataset (hour.csv), Washington, D.C., 2011–2012.")
    st.markdown("**Algorithms compared**\n- Dummy baseline\n- Linear Regression\n- Random Forest\n- Extra Trees\n- Hist Gradient Boosting")
    st.markdown("**NLP feature**\nPlain-English text → entity extraction → editable form → ML prediction")

predict_tab, insights_tab, about_tab = st.tabs(["🔮 Predict demand", "📊 Model insights", "📘 How it works"])
with predict_tab:
    st.subheader("1. Describe your scenario in plain English (optional NLP)")
    st.write("The NLP helper extracts supported values from your sentence and fills the form. Review all values before predicting.")
    st.text_area("Natural-language request", key="nlp_text", height=90)
    if st.button("🧠 Extract details with NLP", use_container_width=False):
        try:
            response = requests.post(f"{API_URL}/parse-text", json={"text": st.session_state["nlp_text"]}, timeout=10)
            if response.ok:
                parsed = response.json()
                st.session_state["nlp_result"] = parsed
                fields = parsed.get("fields", {})
                # Map extracted entities to the Streamlit controls.
                for key in ["season_label", "month", "hour", "weekday_label", "weather_label", "temperature", "feels_like", "humidity", "wind", "holiday_label", "workingday_label", "year_label"]:
                    if key in fields:
                        st.session_state[key] = fields[key]
                st.rerun()
            else:
                st.error(f"NLP API error: {response.text}")
        except requests.RequestException:
            st.error("Cannot connect to the backend. Start FastAPI first; see README.")

    if "nlp_result" in st.session_state:
        parsed = st.session_state["nlp_result"]
        fields = parsed.get("fields", {})
        if fields:
            st.success("NLP extracted: " + ", ".join(parsed.get("entities_found", [])))
            st.json(fields)
        else:
            st.warning(parsed.get("message", "No supported values were extracted."))
        for note in parsed.get("notes", []):
            st.caption("Note: " + note)
        st.caption("Method: rule-based NLP/entity extraction. This is not a separately trained language model.")

    st.divider()
    st.subheader("2. Review or edit the extracted values")
    c1, c2, c3 = st.columns(3)
    with c1:
        season = st.selectbox("Season", ["Spring", "Summer", "Fall", "Winter"], key="season_label")
        season_id = {"Spring": 1, "Summer": 2, "Fall": 3, "Winter": 4}[season]
        month = st.slider("Month", 1, 12, key="month")
        hour = st.slider("Hour (24-hour clock)", 0, 23, key="hour")
        weekday_options = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
        weekday = st.selectbox("Day of week", weekday_options, key="weekday_label")
    with c2:
        weather_options = ["Clear / partly cloudy", "Mist / cloudy", "Light rain / snow", "Heavy rain / snow"]
        weather = st.selectbox("Weather", weather_options, key="weather_label")
        weather_id = {label: i for i, label in enumerate(weather_options, start=1)}[weather]
        temperature = st.slider("Temperature (°C)", -8.0, 39.0, step=0.5, key="temperature")
        feels_like = st.slider("Feels-like temperature (°C)", -16.0, 50.0, step=0.5, key="feels_like")
        humidity = st.slider("Humidity (%)", 0.0, 100.0, step=1.0, key="humidity")
    with c3:
        wind = st.slider("Wind speed (km/h)", 0.0, 100.0, step=1.0, key="wind")
        holiday = st.selectbox("Holiday?", ["No", "Yes"], key="holiday_label")
        workingday = st.selectbox("Working day?", ["Yes", "No"], key="workingday_label")
        year = st.selectbox("Dataset year indicator", ["2012 (1)", "2011 (0)"], key="year_label")

    if st.button("🚲 Predict hourly rentals", type="primary", use_container_width=True):
        body = {
            "season": season_id, "year": 1 if year.startswith("2012") else 0, "month": month, "hour": hour,
            "holiday": 1 if holiday == "Yes" else 0,
            "weekday": weekday_options.index(weekday), "workingday": 1 if workingday == "Yes" else 0,
            "weather": weather_id, "temperature_c": temperature, "feels_like_c": feels_like,
            "humidity_percent": humidity, "wind_speed_kmh": wind,
        }
        try:
            response = requests.post(f"{API_URL}/predict", json=body, timeout=20)
            if response.ok:
                result = response.json()
                st.success("Prediction completed")
                m1, m2 = st.columns(2)
                m1.metric("Estimated rentals in this hour", f"{result['predicted_rentals']:,} bikes")
                m2.metric("Selected model", result["model"])
                st.caption(result["note"])
            else:
                st.error(f"API error: {response.text}")
        except requests.RequestException:
            st.error("Cannot connect to the backend. Start FastAPI in a separate terminal using the README instructions.")

with insights_tab:
    st.subheader("Training results")
    comparison = ROOT / "reports" / "model_comparison.csv"
    metrics_path = ROOT / "reports" / "final_metrics.json"
    if comparison.exists():
        table = pd.read_csv(comparison)
        st.write("Validation metrics — lower MAE/RMSE is better; higher R² is generally better.")
        st.dataframe(table, use_container_width=True, hide_index=True)
        st.bar_chart(table.set_index("Model")[["RMSE", "MAE"]])
    else:
        st.info("Run the training command first. It will generate model comparison reports here.")
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())
        st.markdown(f"**Selected model:** {metrics['selected_model']}")
        st.json(metrics["test_metrics"])
    pred_path = ROOT / "reports" / "test_predictions.csv"
    if pred_path.exists():
        pred = pd.read_csv(pred_path).head(100)
        st.write("First 100 held-out test predictions")
        st.line_chart(pred[["actual", "predicted"]])

with about_tab:
    st.subheader("Project workflow")
    st.markdown("""
    1. **Load:** download and read the official hourly UCI CSV.
    2. **Clean:** parse dates, remove exact duplicates, validate target values and inspect missing data.
    3. **Prepare features:** impute missing values, scale numeric features, one-hot encode categorical features.
    4. **Split chronologically:** earliest 70% for training, next 15% for validation, latest 15% for testing.
    5. **Train:** compare a mean baseline, Linear Regression, Random Forest, Extra Trees and Hist Gradient Boosting.
    6. **Select:** choose the lowest validation RMSE, refit on train + validation, and evaluate once on the held-out test set.
    7. **Save and serve:** persist the full preprocessing/model pipeline; FastAPI provides `/predict` and `/parse-text` endpoints.
    8. **NLP:** rule-based entity extraction recognizes common time, temperature, humidity, season, month, weekday, holiday and weather phrases. The user reviews extracted fields before prediction.

    **Leakage prevention:** `casual` and `registered` are excluded because they add up to `cnt`, the target. The row ID and raw date are not used directly as features.

    **Important:** this dataset is historical (Washington, D.C., 2011–2012), not a live forecast for another city. Text parsing is an educational NLP component, not a trained LLM or a trained NER model.
    """)
