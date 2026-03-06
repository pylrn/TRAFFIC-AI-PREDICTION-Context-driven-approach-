"""
LSTM Traffic Prediction API  (port 8003)

A lightweight 1-layer LSTM (PyTorch) trained at startup on synthetic
multi-step sequences.  It captures temporal patterns — i.e. how a stream
of recent speed / density / weather readings evolves into a predicted
congestion state.

Traffic states
  0 = free_flow  |  1 = moderate  |  2 = heavy  |  3 = jam
"""

import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEQ_LEN    = 6     # number of historical time steps the LSTM sees
N_FEATURES = 5     # speed_norm, density_norm, hour_sin, hour_cos, weather_norm
N_HIDDEN   = 32
N_CLASSES  = 4

WEATHER_MAP: dict[str, float] = {
    "clear": 0.0, "sunny": 0.0, "cloudy": 0.2,
    "fog": 0.4,   "rain": 0.7,  "snow": 1.0,
}


def _encode_step(speed: float, count: int, hour: int, weather: str) -> list[float]:
    """Normalise a single time step into a fixed-length feature vector."""
    hour_sin = float(np.sin(2 * np.pi * hour / 24))
    hour_cos = float(np.cos(2 * np.pi * hour / 24))
    return [
        min(speed / 80.0, 1.0),
        min(count / 300.0, 1.0),
        hour_sin,
        hour_cos,
        WEATHER_MAP.get(weather.lower(), 0.2),
    ]


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------
class LSTMTrafficModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lstm = nn.LSTM(N_FEATURES, N_HIDDEN, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(N_HIDDEN, N_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, T, F)
        _, (h_n, _) = self.lstm(x)
        return self.fc(self.dropout(h_n[-1]))


# ---------------------------------------------------------------------------
# Synthetic training sequences
# ---------------------------------------------------------------------------
def _generate_sequences(n: int = 600) -> tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(7)
    weathers = list(WEATHER_MAP.keys())
    X_all, y_all = [], []

    for _ in range(n):
        speed_base = float(rng.uniform(10, 75))
        count_base = int(rng.integers(30, 300))
        hour       = int(rng.integers(0, 24))

        seq = []
        for _ in range(SEQ_LEN):
            s = float(np.clip(speed_base + rng.normal(0, 4), 5, 80))
            c = int(np.clip(count_base + rng.integers(-20, 20), 10, 350))
            w = str(rng.choice(weathers))
            seq.append(_encode_step(s, c, hour, w))

        congestion = (1.0 - speed_base / 80.0) * 0.50 + (count_base / 300.0) * 0.50
        label = 0 if congestion < 0.25 else (
                1 if congestion < 0.50 else (
                2 if congestion < 0.75 else 3))

        X_all.append(seq)
        y_all.append(label)

    return (
        torch.tensor(X_all, dtype=torch.float32),
        torch.tensor(y_all,  dtype=torch.long),
    )


# ---------------------------------------------------------------------------
# Train at startup   (~1–2 s on CPU)
# ---------------------------------------------------------------------------
_X_seq, _y_seq = _generate_sequences()

LSTM_MODEL = LSTMTrafficModel()
_optimiser  = torch.optim.Adam(LSTM_MODEL.parameters(), lr=1e-3)
_criterion  = nn.CrossEntropyLoss()

LSTM_MODEL.train()
for _epoch in range(40):
    _optimiser.zero_grad()
    _loss = _criterion(LSTM_MODEL(_X_seq), _y_seq)
    _loss.backward()
    _optimiser.step()

LSTM_MODEL.eval()

# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
app = FastAPI(title="LSTM Model API", version="1.0.0")


class PredictRequest(BaseModel):
    road_id: int
    avg_speed: float
    vehicle_count: int
    weather: str
    hour: int
    previous_state: int


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model": "lstm"}


@app.post("/predict/lstm")
def predict(payload: PredictRequest) -> dict:
    # Build an artificial SEQ_LEN-step sequence from the single snapshot.
    # Small jitter simulates recent history variation.
    step = _encode_step(payload.avg_speed, payload.vehicle_count, payload.hour, payload.weather)
    rng  = np.random.default_rng()
    seq = [
        [
            float(np.clip(step[0] + rng.normal(0, 0.03), 0, 1)),
            float(np.clip(step[1] + rng.normal(0, 0.03), 0, 1)),
            step[2], step[3], step[4],
        ]
        for _ in range(SEQ_LEN)
    ]

    x_tensor = torch.tensor([seq], dtype=torch.float32)
    with torch.no_grad():
        logits = LSTM_MODEL(x_tensor)
        probs  = torch.softmax(logits, dim=-1).squeeze().tolist()

    return {"model": "lstm", "probabilities": probs}
