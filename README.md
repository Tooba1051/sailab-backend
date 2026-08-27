# Sailab Flood Risk API

FastAPI backend that serves flood risk predictions from `flood_risk_pipeline.joblib`.

## 1. Set up locally

```bash
# create + activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # on Windows: venv\Scripts\activate

# install exact dependency versions (scikit-learn MUST be 1.6.1)
pip install -r requirements.txt
```

Place your trained model file in this same folder as `flood_risk_pipeline.joblib`
(this file is NOT included here — copy it in yourself).

## 2. Run it

```bash
uvicorn main:app --reload --port 8000
```

Then open http://127.0.0.1:8000/docs — FastAPI gives you an interactive
Swagger UI where you can test /predict directly in the browser before
touching Flutter at all.

## 3. Test a prediction

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "latitude": 31.5204,
    "longitude": 74.3587,
    "elevation_m": 213,
    "rainfall_mm": 45.2,
    "distance_to_river_km": 3.1,
    "rain_24h": 12,
    "rain_3day": 38.5,
    "rain_7day": 95,
    "temperature_c": 29.4,
    "humidity_pct": 78,
    "province": "Punjab"
  }'
```

Expected shape back:

```json
{ "percent_flooded": 2.43, "risk_level": "Low" }
```

## 4. Deploy so the phone app can reach it

Once it works locally, deploy to Render or Railway (both have free tiers
and handle Python services well):

- Push this folder to a GitHub repo
- On Render/Railway: "New Web Service" → connect the repo
- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- Upload `flood_risk_pipeline.joblib` alongside the code (or load it from
  cloud storage if it's large)

You'll get a public URL like `https://sailab-api.onrender.com` — that's
what the Flutter app calls instead of `127.0.0.1`.

## Notes

- `scikit-learn==1.6.1` is pinned deliberately — the pipeline was trained/saved
  under that version and will fail to unpickle on newer sklearn versions.
- CORS is wide open (`allow_origins=["*"]`) for development. Tighten this
  once you know your app's real domain/bundle id.
- `risk_level_from_percent()` thresholds (Low/Moderate/High/Severe) are a
  starting guess — adjust once you've seen real prediction ranges.
