"""
Bayesian Network Traffic Prediction API  (port 8004)

Models the probabilistic relationships:

    Weather ──┐
  TimeOfDay ──┼──► CongestionLevel
    Density ──┘

Conditional probability tables are hand-crafted from domain knowledge.
In production these would be learned from historical data with MLE/EM.

Traffic states
  0 = free_flow  |  1 = moderate  |  2 = heavy  |  3 = jam
"""

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from pgmpy.models import DiscreteBayesianNetwork

# ---------------------------------------------------------------------------
# Discretisation helpers
# ---------------------------------------------------------------------------
def _discretize_weather(w: str) -> int:
    """0=clear  1=adverse(fog/rain)  2=severe(snow)"""
    mapping = {
        "clear": 0, "sunny": 0, "cloudy": 0,
        "fog":   1, "rain":  1,
        "snow":  2,
    }
    return mapping.get(w.lower(), 0)


def _discretize_time(hour: int) -> int:
    """0=night(00-05)  1=normal  2=rush(07-10, 17-20)"""
    if 0 <= hour <= 5:
        return 0
    if (7 <= hour <= 10) or (17 <= hour <= 20):
        return 2
    return 1


def _discretize_density(count: int) -> int:
    """0=low(<100)  1=medium(100-200)  2=high(>200)"""
    if count < 100:
        return 0
    if count < 200:
        return 1
    return 2


# ---------------------------------------------------------------------------
# Build Bayesian Network
# ---------------------------------------------------------------------------
def _build_network() -> VariableElimination:
    model = DiscreteBayesianNetwork(
        [
            ("Weather",   "Congestion"),
            ("TimeOfDay", "Congestion"),
            ("Density",   "Congestion"),
        ]
    )

    # Prior distributions
    cpd_weather   = TabularCPD("Weather",   3, [[0.65], [0.25], [0.10]])
    cpd_time      = TabularCPD("TimeOfDay", 3, [[0.25], [0.50], [0.25]])
    cpd_density   = TabularCPD("Density",   3, [[0.35], [0.40], [0.25]])

    # P(Congestion | Weather, TimeOfDay, Density)
    # 3 parents × 3 states each = 27 parent combinations → (4, 27) CPT
    # Iteration order: Weather varies slowest, Density varies fastest
    cpt_cols: list[list[float]] = []
    for weather in range(3):    # 0=clear, 1=adverse, 2=severe
        for time in range(3):   # 0=night, 1=normal, 2=rush
            for density in range(3):  # 0=low, 1=med, 2=high
                base = np.array([0.60, 0.25, 0.10, 0.05], dtype=float)

                # Weather
                weather_mods = [(1.00, 1.00), (0.70, 1.50), (0.45, 2.50)]
                base[0] *= weather_mods[weather][0]
                base[3] *= weather_mods[weather][1]

                # Time of day
                time_mods = [(1.25, 0.70, 0.55), (1.00, 1.00, 1.00), (0.55, 1.40, 1.90)]
                base[0] *= time_mods[time][0]
                base[2] *= time_mods[time][1]
                base[3] *= time_mods[time][2]

                # Vehicle density
                density_mods = [(1.45, 0.60, 0.50), (1.00, 1.00, 1.00), (0.45, 1.55, 2.10)]
                base[0] *= density_mods[density][0]
                base[2] *= density_mods[density][1]
                base[3] *= density_mods[density][2]

                base = np.clip(base, 1e-6, None)
                base /= base.sum()
                cpt_cols.append(base.tolist())

    # TabularCPD expects shape (n_states_child, n_parent_combinations)
    cpt_matrix = np.array(cpt_cols).T  # → (4, 27)

    cpd_congestion = TabularCPD(
        variable="Congestion",
        variable_card=4,
        values=cpt_matrix,
        evidence=["Weather", "TimeOfDay", "Density"],
        evidence_card=[3, 3, 3],
    )

    model.add_cpds(cpd_weather, cpd_time, cpd_density, cpd_congestion)
    assert model.check_model(), "Bayesian Network CPDs are invalid"

    return VariableElimination(model)


_INFERENCE = _build_network()

# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
app = FastAPI(title="Bayesian Network Model API", version="1.0.0")


class PredictRequest(BaseModel):
    road_id: int
    avg_speed: float
    vehicle_count: int
    weather: str
    hour: int
    previous_state: int


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model": "bayesian"}


@app.post("/predict/bayesian")
def predict(payload: PredictRequest) -> dict:
    evidence = {
        "Weather":   _discretize_weather(payload.weather),
        "TimeOfDay": _discretize_time(payload.hour),
        "Density":   _discretize_density(payload.vehicle_count),
    }
    result = _INFERENCE.query(variables=["Congestion"], evidence=evidence, show_progress=False)
    probs = result.values.tolist()
    return {"model": "bayesian", "probabilities": probs}
