"""
Random Forest Traffic Prediction API  (port 8002)

Trained at startup on synthetic data that encodes domain knowledge about
speed, vehicle density, time-of-day, and weather.  In production this
module would load a serialised model from disk instead.

Traffic states
  0 = free_flow  |  1 = moderate  |  2 = heavy  |  3 = jam
"""

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.ensemble import RandomForestClassifier

# ---------------------------------------------------------------------------
# Weather encoding
# ---------------------------------------------------------------------------
WEATHER_MAP: dict[str, int] = {
    "clear": 0, "sunny": 0,
    "cloudy": 1,
    "fog": 2,
    "rain": 3,
    "snow": 4,
}


def _encode_weather(w: str) -> int:
    return WEATHER_MAP.get(w.lower(), 1)


# ---------------------------------------------------------------------------
# Synthetic training data (reflects domain rules)
# ---------------------------------------------------------------------------
def _generate_data(n: int = 1000) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    weathers = list(WEATHER_MAP.keys())
    X, y = [], []

    for _ in range(n):
        speed   = float(rng.uniform(5, 80))
        count   = int(rng.integers(20, 350))
        hour    = int(rng.integers(0, 24))
        weather = str(rng.choice(weathers))
        prev    = int(rng.integers(0, 4))

        congestion  = max(0.0, (60.0 - speed) / 60.0) * 0.40
        congestion += min(count / 300.0, 1.0)            * 0.30
        congestion += 0.20 if (7 <= hour <= 9 or 17 <= hour <= 19) else 0.0
        congestion += {
            "clear": 0.00, "sunny": 0.00, "cloudy": 0.05,
            "fog": 0.10,   "rain": 0.20,  "snow": 0.35,
        }.get(weather, 0.05)
        congestion += prev * 0.04
        congestion  = min(congestion, 1.0)

        label = 0 if congestion < 0.25 else (
                1 if congestion < 0.50 else (
                2 if congestion < 0.75 else 3))

        X.append([speed, count, hour, _encode_weather(weather), prev])
        y.append(label)

    return np.array(X, dtype=float), np.array(y, dtype=int)


_X_train, _y_train = _generate_data()

RF_MODEL = RandomForestClassifier(
    n_estimators=150, max_depth=10, min_samples_leaf=3, random_state=42
)
RF_MODEL.fit(_X_train, _y_train)

# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
app = FastAPI(title="Random Forest Model API", version="1.0.0")


class PredictRequest(BaseModel):
    road_id: int
    avg_speed: float
    vehicle_count: int
    weather: str
    hour: int
    previous_state: int


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model": "random_forest"}


@app.post("/predict/randomforest")
def predict(payload: PredictRequest) -> dict:
    features = np.array([[
        payload.avg_speed,
        payload.vehicle_count,
        payload.hour,
        _encode_weather(payload.weather),
        payload.previous_state,
    ]], dtype=float)

    raw_probs = RF_MODEL.predict_proba(features)[0]

    # Align with all 4 classes in case some were absent during training
    full_probs = np.zeros(4, dtype=float)
    for cls_idx, cls_val in enumerate(RF_MODEL.classes_):
        full_probs[int(cls_val)] = raw_probs[cls_idx]

    full_probs = np.clip(full_probs, 1e-9, None)
    full_probs /= full_probs.sum()

    return {"model": "random_forest", "probabilities": full_probs.tolist()}
