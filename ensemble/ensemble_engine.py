"""
Ensemble Engine

Holds context-specific model weights and performs weighted probability fusion
over four traffic states: free_flow, moderate, heavy, jam.
"""

from dataclasses import dataclass

import numpy as np

from ensemble.context_detector import TrafficContext

# ---------------------------------------------------------------------------
# Traffic state labels
# ---------------------------------------------------------------------------
TRAFFIC_STATES = ["free", "moderate", "heavy", "jam"]

# ---------------------------------------------------------------------------
# Context → model weights  (must sum to 1.0)
# ---------------------------------------------------------------------------
CONTEXT_WEIGHTS: dict[TrafficContext, dict[str, float]] = {
    TrafficContext.MORNING_RUSH: {
        "lstm": 0.5, "random_forest": 0.2, "markov": 0.2, "bayesian": 0.1,
    },
    TrafficContext.EVENING_RUSH: {
        "lstm": 0.5, "random_forest": 0.2, "markov": 0.2, "bayesian": 0.1,
    },
    TrafficContext.NIGHT_LOW_TRAFFIC: {
        "markov": 0.5, "random_forest": 0.3, "lstm": 0.1, "bayesian": 0.1,
    },
    TrafficContext.WEATHER_EVENT: {
        "bayesian": 0.5, "random_forest": 0.2, "lstm": 0.2, "markov": 0.1,
    },
    TrafficContext.ACCIDENT_EVENT: {
        "random_forest": 0.5, "lstm": 0.2, "bayesian": 0.2, "markov": 0.1,
    },
    TrafficContext.NORMAL_CONDITIONS: {
        "random_forest": 0.4, "lstm": 0.3, "markov": 0.2, "bayesian": 0.1,
    },
}


# ---------------------------------------------------------------------------
# Data container for the four model outputs
# ---------------------------------------------------------------------------
@dataclass
class ModelPredictions:
    markov:        list[float]   # [P_free, P_moderate, P_heavy, P_jam]
    random_forest: list[float]
    lstm:          list[float]
    bayesian:      list[float]


# ---------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------
def fuse(context: TrafficContext, predictions: ModelPredictions) -> dict:
    """
    Compute the weighted-average probability distribution and derive
    the predicted state and confidence.

    Confidence = sum of the two highest class probabilities (top-2),
    capped at 1.0.  It expresses how much probability mass is concentrated
    around the most likely outcome.
    """
    weights = CONTEXT_WEIGHTS[context]
    model_map: dict[str, list[float]] = {
        "markov":        predictions.markov,
        "random_forest": predictions.random_forest,
        "lstm":          predictions.lstm,
        "bayesian":      predictions.bayesian,
    }

    final = np.zeros(4, dtype=float)
    for model_name, w in weights.items():
        final += w * np.array(model_map[model_name], dtype=float)

    final = np.clip(final, 1e-9, None)
    final /= final.sum()

    predicted_idx: int = int(np.argmax(final))
    sorted_probs = np.sort(final)[::-1]
    confidence = float(min(sorted_probs[0] + sorted_probs[1], 1.0))

    return {
        "final_probabilities": {
            state: round(float(p), 4)
            for state, p in zip(TRAFFIC_STATES, final)
        },
        "predicted_state": TRAFFIC_STATES[predicted_idx],
        "confidence":      round(confidence, 4),
        "model_weights":   weights,
    }
