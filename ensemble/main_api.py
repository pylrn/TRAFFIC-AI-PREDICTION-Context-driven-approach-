"""
Traffic Ensemble Orchestrator API  (port 8000)

Exposes:
    GET  /                  – interactive web UI for non-technical users
    POST /predict_traffic   – JSON prediction endpoint (with NL summary)
    GET  /health            – liveness check
"""

import asyncio
import os
from datetime import datetime, timezone
from typing import Optional

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from ensemble.context_detector import TrafficContext, detect_context
from ensemble.ensemble_engine import ModelPredictions, fuse
from ensemble.tomtom_fetcher import fetch_traffic_for_location

load_dotenv()

_STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

# ---------------------------------------------------------------------------
# Model service URLs
# ---------------------------------------------------------------------------
MODEL_URLS: dict[str, str] = {
    "markov":        "http://localhost:8001/predict/markov",
    "random_forest": "http://localhost:8002/predict/randomforest",
    "lstm":          "http://localhost:8003/predict/lstm",
    "bayesian":      "http://localhost:8004/predict/bayesian",
}

# ---------------------------------------------------------------------------
# Natural language summary generator
# ---------------------------------------------------------------------------
_STATE_PHRASES: dict[str, str] = {
    "free":     "traffic is flowing freely — roads are clear",
    "moderate": "traffic is moderately congested — expect minor slowdowns",
    "heavy":    "traffic is heavily congested — significant delays likely",
    "jam":      "there is a severe traffic jam — roads are nearly at a standstill",
}

_CONTEXT_PHRASES: dict[TrafficContext, str] = {
    TrafficContext.MORNING_RUSH:      "morning rush hour",
    TrafficContext.EVENING_RUSH:      "evening rush hour",
    TrafficContext.NIGHT_LOW_TRAFFIC: "late-night low-traffic hours",
    TrafficContext.WEATHER_EVENT:     "adverse weather conditions",
    TrafficContext.ACCIDENT_EVENT:    "an active incident on the road",
    TrafficContext.NORMAL_CONDITIONS: "normal daytime conditions",
}

_MODEL_NAMES: dict[str, str] = {
    "lstm":          "LSTM Neural Network",
    "random_forest": "Random Forest",
    "markov":        "Markov Chain",
    "bayesian":      "Bayesian Network",
}

_MODEL_STRENGTHS: dict[str, str] = {
    "lstm":          "best at detecting patterns in recent traffic history",
    "random_forest": "best at handling complex speed and density relationships",
    "markov":        "best at predicting steady low-traffic transitions",
    "bayesian":      "best at reasoning about weather and environmental factors",
}


def _generate_nl_summary(
    road_id:         int,
    hour:            int,
    weather:         str,
    avg_speed:       float,
    vehicle_count:   int,
    accident:        bool,
    context:         TrafficContext,
    predicted_state: str,
    confidence:      float,
    final_probs:     dict[str, float],
    model_weights:   dict[str, float],
    road_name:       str | None = None,
) -> str:
    hour_12    = hour % 12 or 12
    am_pm      = "AM" if hour < 12 else "PM"
    time_str   = f"{hour_12}:00 {am_pm}"
    road_label = road_name if road_name else f"Road {road_id}"

    conf_pct  = int(confidence * 100)
    conf_word = (
        "very confident" if confidence >= 0.80 else
        "fairly confident" if confidence >= 0.60 else
        "somewhat uncertain"
    )

    lead_model    = max(model_weights, key=model_weights.get)  # type: ignore[arg-type]
    lead_pct      = int(model_weights[lead_model] * 100)
    second_best   = sorted(final_probs, key=final_probs.get, reverse=True)[1]  # type: ignore[arg-type]
    second_pct    = int(final_probs[second_best] * 100)

    incident_note = " An accident has been reported on this road." if accident else ""
    weather_note  = (
        f" Weather is {weather}, which is affecting road conditions."
        if weather.lower() in ("rain", "snow", "fog") else ""
    )

    summary = (
        f"At {time_str}, {road_label} is experiencing {_CONTEXT_PHRASES[context]}.{incident_note}{weather_note} "
        f"Current average speed is {avg_speed:.0f} km/h with {vehicle_count} vehicles observed. "
        f"The AI predicts that {_STATE_PHRASES[predicted_state]}. "
        f"The system is {conf_word} about this ({conf_pct}% confidence). "
        f"The {_MODEL_NAMES[lead_model]} had the highest influence on this prediction ({lead_pct}% weight) "
        f"because it is {_MODEL_STRENGTHS[lead_model]}. "
        f"There is also a {second_pct}% chance of {second_best} traffic conditions."
    )
    return summary


# ---------------------------------------------------------------------------
# Interactive HTML UI
# ---------------------------------------------------------------------------
_HTML_UI = ""  # now served from ensemble/static/index.html

# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------
class TrafficRequest(BaseModel):
    road_id:        int           = Field(..., description="Unique road / segment identifier")
    timestamp:      Optional[str] = Field(None, description="ISO-8601; defaults to now (UTC)")
    avg_speed:      float         = Field(..., ge=0, description="Average speed (km/h)")
    vehicle_count:  int           = Field(..., ge=0, description="Vehicles counted in observation window")
    weather:        str           = Field("clear", description="clear | sunny | cloudy | fog | rain | snow")
    accident:       bool          = Field(False, description="Active accident on this segment")
    hour:           int           = Field(..., ge=0, le=23, description="Hour of day (0-23)")
    day_of_week:    Optional[int] = Field(None, ge=0, le=6, description="0=Mon … 6=Sun")
    previous_state: int           = Field(..., ge=0, le=3,
                                          description="0=free_flow 1=moderate 2=heavy 3=jam")


class PredictionResponse(BaseModel):
    road_id:                  int
    timestamp:                str
    context:                  str
    model_predictions:        dict[str, list[float]]
    model_weights:            dict[str, float]
    final_probabilities:      dict[str, float]
    predicted_state:          str
    confidence:               float
    natural_language_summary: str


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Traffic Ensemble API",
    version="1.0.0",
    description="Context-aware probabilistic ensemble traffic prediction system",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
async def _call_model(
    client:  httpx.AsyncClient,
    name:    str,
    payload: dict,
) -> tuple[str, list[float]]:
    try:
        resp = await client.post(MODEL_URLS[name], json=payload, timeout=8.0)
        resp.raise_for_status()
        return name, resp.json()["probabilities"]
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Model '{name}' returned HTTP {exc.response.status_code}",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Model '{name}' unreachable: {exc}",
        )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def ui() -> HTMLResponse:
    html_path = os.path.join(_STATIC_DIR, "index.html")
    return HTMLResponse(open(html_path, encoding="utf-8").read())


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "ensemble"}


@app.post("/predict_traffic", response_model=PredictionResponse)
async def predict_traffic(req: TrafficRequest) -> PredictionResponse:
    ts: str = req.timestamp or datetime.now(timezone.utc).isoformat()
    context: TrafficContext = detect_context(req.hour, req.weather, req.accident)

    model_payload = {
        "road_id":        req.road_id,
        "avg_speed":      req.avg_speed,
        "vehicle_count":  req.vehicle_count,
        "weather":        req.weather,
        "hour":           req.hour,
        "previous_state": req.previous_state,
    }

    async with httpx.AsyncClient() as client:
        results: list[tuple[str, list[float]]] = await asyncio.gather(
            _call_model(client, "markov",        model_payload),
            _call_model(client, "random_forest", model_payload),
            _call_model(client, "lstm",          model_payload),
            _call_model(client, "bayesian",      model_payload),
        )

    predictions_dict: dict[str, list[float]] = dict(results)

    preds = ModelPredictions(
        markov=        predictions_dict["markov"],
        random_forest= predictions_dict["random_forest"],
        lstm=          predictions_dict["lstm"],
        bayesian=      predictions_dict["bayesian"],
    )

    fusion = fuse(context, preds)
    nl     = _generate_nl_summary(
        road_id=         req.road_id,
        hour=            req.hour,
        weather=         req.weather,
        avg_speed=       req.avg_speed,
        vehicle_count=   req.vehicle_count,
        accident=        req.accident,
        context=         context,
        predicted_state= fusion["predicted_state"],
        confidence=      fusion["confidence"],
        final_probs=     fusion["final_probabilities"],
        model_weights=   fusion["model_weights"],
    )

    return PredictionResponse(
        road_id=                  req.road_id,
        timestamp=                ts,
        context=                  context.value,
        model_predictions=        predictions_dict,
        model_weights=            fusion["model_weights"],
        final_probabilities=      fusion["final_probabilities"],
        predicted_state=          fusion["predicted_state"],
        confidence=               fusion["confidence"],
        natural_language_summary= nl,
    )


# ---------------------------------------------------------------------------
# Map endpoints
# ---------------------------------------------------------------------------
@app.get("/fetch_location")
async def fetch_location(lat: float, lon: float) -> dict:
    """Return raw live traffic + weather for a lat/lon (from map click)."""
    return await fetch_traffic_for_location(lat, lon)


@app.post("/predict_from_map")
async def predict_from_map(lat: float, lon: float) -> dict:
    """
    One-click prediction from a map point:
    1. Fetch live data via TomTom + Open-Meteo.
    2. Run the full ensemble prediction.
    3. Return combined result with road name and live input.
    """
    live = await fetch_traffic_for_location(lat, lon)

    req = TrafficRequest(
        road_id=        live["road_id"],
        avg_speed=      live["avg_speed"],
        vehicle_count=  live["vehicle_count"],
        weather=        live["weather"],
        accident=       live["accident"],
        hour=           live["hour"],
        day_of_week=    live.get("day_of_week"),
        previous_state= live["previous_state"],
    )

    result = await predict_traffic(req)

    # Overwrite NL summary to include the real road name
    nl_with_name = _generate_nl_summary(
        road_id=         live["road_id"],
        hour=            live["hour"],
        weather=         live["weather"],
        avg_speed=       live["avg_speed"],
        vehicle_count=   live["vehicle_count"],
        accident=        live["accident"],
        context=         detect_context(live["hour"], live["weather"], live["accident"]),
        predicted_state= result.predicted_state,
        confidence=      result.confidence,
        final_probs=     result.final_probabilities,
        model_weights=   result.model_weights,
        road_name=       live.get("road_name"),
    )

    return {
        **result.model_dump(),
        "natural_language_summary": nl_with_name,
        "road_name":  live.get("road_name", "Unknown Road"),
        "live_input": live,
    }
