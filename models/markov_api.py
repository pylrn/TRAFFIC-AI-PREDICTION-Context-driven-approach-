"""
Markov Chain Traffic Prediction API  (port 8001)

Predicts the next-state probability distribution using a 4×4 transition
matrix that is then adjusted by real-time speed, density, and weather.

Traffic states
  0 = free_flow  |  1 = moderate  |  2 = heavy  |  3 = jam
"""

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Transition matrix  P(next | current)
# Rows = current state, Columns = next state
# ---------------------------------------------------------------------------
TRANSITION_MATRIX = np.array(
    [
        [0.70, 0.20, 0.08, 0.02],  # from free_flow
        [0.15, 0.55, 0.25, 0.05],  # from moderate
        [0.05, 0.20, 0.55, 0.20],  # from heavy
        [0.02, 0.08, 0.30, 0.60],  # from jam
    ],
    dtype=float,
)

app = FastAPI(title="Markov Model API", version="1.0.0")


class PredictRequest(BaseModel):
    road_id: int
    avg_speed: float
    vehicle_count: int
    weather: str
    hour: int
    previous_state: int  # 0-3


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model": "markov"}


@app.post("/predict/markov")
def predict(payload: PredictRequest) -> dict:
    state = max(0, min(3, payload.previous_state))
    probs = TRANSITION_MATRIX[state].copy()

    # Higher speed → push probability toward free flow, away from jam
    speed_ratio = min(payload.avg_speed / 80.0, 1.0)
    probs[0] *= 1.0 + 0.40 * speed_ratio
    probs[3] *= max(0.1, 1.0 - 0.50 * speed_ratio)

    # Higher vehicle count → push toward congested states
    density_ratio = min(payload.vehicle_count / 300.0, 1.0)
    probs[2] *= 1.0 + 0.30 * density_ratio
    probs[3] *= 1.0 + 0.40 * density_ratio

    # Adverse weather → suppress free-flow, amplify jam
    if payload.weather.lower() in ("rain", "snow", "fog"):
        probs[0] *= 0.65
        probs[3] *= 1.40

    probs = np.clip(probs, 1e-9, None)
    probs /= probs.sum()

    return {"model": "markov", "probabilities": probs.tolist()}
