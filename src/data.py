"""Download and prepare the official UCI Bike Sharing hourly dataset."""
from pathlib import Path
import io
import zipfile
import urllib.request
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
CSV_PATH = RAW_DIR / "hour.csv"
DATA_URL = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"


def download_dataset(force: bool = False) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if CSV_PATH.exists() and not force:
        return CSV_PATH
    print("Downloading the official UCI Bike Sharing Dataset...")
    try:
        with urllib.request.urlopen(DATA_URL, timeout=45) as response:
            payload = response.read()
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            member = next((n for n in archive.namelist() if n.lower().endswith("hour.csv")), None)
            if member is None:
                raise FileNotFoundError("hour.csv was not found inside the UCI archive")
            CSV_PATH.write_bytes(archive.read(member))
    except Exception as exc:
        raise RuntimeError(
            "Could not download the dataset. Check your internet connection and run again, "
            "or manually download the UCI Bike Sharing ZIP from "
            "https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset and place hour.csv in data/raw/."
        ) from exc
    print(f"Dataset saved to {CSV_PATH}")
    return CSV_PATH


def load_data() -> pd.DataFrame:
    path = download_dataset()
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    required = {"dteday", "hr", "season", "yr", "mnth", "holiday", "weekday", "workingday", "weathersit", "temp", "atemp", "hum", "windspeed", "cnt"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {sorted(missing)}")
    df["dteday"] = pd.to_datetime(df["dteday"], errors="coerce")
    df = df.dropna(subset=["dteday", "cnt"]).drop_duplicates()
    df = df.sort_values(["dteday", "hr"]).reset_index(drop=True)
    # Remove invalid target rows; this dataset is expected to have no missing values.
    df = df[df["cnt"].ge(0)].copy()
    return df
