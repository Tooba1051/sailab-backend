"""
Sailab — Flood Risk Prediction API
FastAPI backend that loads the trained sklearn/XGBoost pipeline
and exposes a /predict endpoint for the Flutter app.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Literal
import joblib
import pandas as pd
import os

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Sailab Flood Risk API",
    description="Predicts percent-flooded risk for a given location and weather profile.",
    version="1.0.0",
)

# Allow the Flutter app (and local dev / web builds) to call this API.
# Tighten allow_origins to your real domains once you deploy the app.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "flood_risk_pipeline.joblib")

# The pipeline was saved with scikit-learn 1.6.1 — this service's
# requirements.txt pins that exact version so it unpickles cleanly.
pipeline = None


@app.on_event("startup")
def load_model():
    global pipeline
    try:
        pipeline = joblib.load(MODEL_PATH)
    except Exception as e:
        # Fail loudly at startup rather than on the first request.
        raise RuntimeError(f"Failed to load model at {MODEL_PATH}: {e}")


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

PROVINCES = Literal[
    "AzadKashmir",
    "Balochistan",
    "FederallyAdministeredTribalAr",
    "Gilgit-Baltistan",
    "Islamabad",
    "Khyber-Pakhtunkhwa",
    "Punjab",
    "Sindh",
    "Unknown",
]


class PredictRequest(BaseModel):
    latitude: float = Field(..., example=31.5204)
    longitude: float = Field(..., example=74.3587)
    elevation_m: float = Field(..., example=213.0)
    rainfall_mm: float = Field(..., example=45.2)
    distance_to_river_km: float = Field(..., example=3.1)
    rain_24h: float = Field(..., example=12.0)
    rain_3day: float = Field(..., example=38.5)
    rain_7day: float = Field(..., example=95.0)
    temperature_c: float = Field(..., example=29.4)
    humidity_pct: float = Field(..., example=78.0)
    province: PROVINCES = Field(..., example="Punjab")


class PredictResponse(BaseModel):
    percent_flooded: float
    risk_level: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Feature order the pipeline expects — must match training exactly.
FEATURE_ORDER = [
    "latitude",
    "longitude",
    "elevation_m",
    "rainfall_mm",
    "distance_to_river_km",
    "rain_24h",
    "rain_3day",
    "rain_7day",
    "temperature_c",
    "humidity_pct",
    "province",
]


def risk_level_from_percent(pct: float) -> str:
    """Simple bucketing for the app's risk badge. Tune thresholds as you validate the model."""
    if pct < 5:
        return "Low"
    elif pct < 20:
        return "Moderate"
    elif pct < 50:
        return "High"
    else:
        return "Severe"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    return {"status": "ok", "service": "Sailab Flood Risk API"}


@app.get("/health")
def health():
    return {"model_loaded": pipeline is not None}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    row = {field: getattr(req, field) for field in FEATURE_ORDER}
    df = pd.DataFrame([row], columns=FEATURE_ORDER)

    try:
        prediction = pipeline.predict(df)[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    pct = float(prediction)
    return PredictResponse(
        percent_flooded=round(pct, 2),
        risk_level=risk_level_from_percent(pct),
    )
