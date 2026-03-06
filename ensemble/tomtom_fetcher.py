"""
TomTom + Open-Meteo Live Data Fetcher

Given a lat/lon from a map click this module:
  1. Calls TomTom Flow Segment API  → live speed, free-flow speed
  2. Calls Open-Meteo              → current weather (free, no key needed)
  3. Calls TomTom Reverse Geocode  → road / street name
  4. Returns a dict ready to feed into TrafficRequest

Graceful degradation:
  If TOMTOM_API_KEY is not set the module falls back to Open-Meteo for
  weather and generates plausible traffic data from the lat/lon + hour so
  the UI still works for demos without a paid key.
"""

import hashlib
import math
import os
from datetime import datetime, timezone

import httpx


# ---------------------------------------------------------------------------
# WMO weather-code → our label
# ---------------------------------------------------------------------------
def _wmo_to_label(code: int) -> str:
    if code == 0:               return "sunny"
    if code in (1, 2, 3):       return "cloudy"
    if code in (45, 48):        return "fog"
    if code in range(51, 68):   return "rain"
    if code in range(71, 78):   return "snow"
    if code in range(80, 83):   return "rain"
    if code in range(95, 100):  return "rain"
    return "clear"


# ---------------------------------------------------------------------------
# Deterministic synthetic fallback (no API key)
# ---------------------------------------------------------------------------
def _synthetic_data(lat: float, lon: float, hour: int) -> dict:
    """
    Produce plausible but fake traffic data so the map UI works without
    a TomTom key.  Values are seeded by lat/lon so the same point always
    returns the same base values.
    """
    seed = int(hashlib.md5(f"{lat:.3f}{lon:.3f}".encode()).hexdigest(), 16)
    rng_speed  = 20 + (seed % 60)            # 20–79 km/h
    rng_count  = 50 + (seed % 250)           # 50–299
    rush       = (7 <= hour <= 9) or (17 <= hour <= 19)
    if rush:
        rng_speed  = max(10, rng_speed - 25)
        rng_count  = min(350, rng_count + 100)
    ff_speed = 80
    ratio    = rng_speed / ff_speed
    prev     = 0 if ratio > 0.8 else (1 if ratio > 0.6 else (2 if ratio > 0.35 else 3))
    road_id  = abs(seed) % 100000
    return {
        "road_id":           road_id,
        "road_name":         f"Road {road_id} (simulated)",
        "lat":               lat,
        "lon":               lon,
        "avg_speed":         float(rng_speed),
        "vehicle_count":     rng_count,
        "weather":           "clear",
        "hour":              hour,
        "day_of_week":       datetime.now(timezone.utc).weekday(),
        "previous_state":    prev,
        "accident":          False,
        "free_flow_speed":   ff_speed,
        "tomtom_confidence": 0.5,
        "data_source":       "simulated",
    }


# ---------------------------------------------------------------------------
# Main fetcher
# ---------------------------------------------------------------------------
async def fetch_traffic_for_location(lat: float, lon: float) -> dict:
    """
    Async: fetch live TomTom + Open-Meteo data for a map point.
    Falls back to synthetic data if TOMTOM_API_KEY is missing.
    """
    api_key = os.getenv("TOMTOM_API_KEY", "").strip()
    now     = datetime.now(timezone.utc)
    hour    = now.hour

    # ------------------------------------------------------------------
    # Open-Meteo weather (always free, no key)
    # ------------------------------------------------------------------
    weather_label = "clear"
    try:
        async with httpx.AsyncClient(timeout=6) as client:
            w_resp = await client.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude":        lat,
                    "longitude":       lon,
                    "current_weather": "true",
                },
            )
            w_data = w_resp.json()
            code   = w_data.get("current_weather", {}).get("weathercode", 0)
            weather_label = _wmo_to_label(code)
    except Exception:
        pass  # weather stays "clear"

    # ------------------------------------------------------------------
    # No TomTom key → synthetic fallback
    # ------------------------------------------------------------------
    if not api_key:
        result = _synthetic_data(lat, lon, hour)
        result["weather"]     = weather_label
        result["data_source"] = "simulated (no TOMTOM_API_KEY)"
        return result

    # ------------------------------------------------------------------
    # TomTom Flow Segment
    # ------------------------------------------------------------------
    try:
        async with httpx.AsyncClient(timeout=8) as client:
            # Flow
            flow_resp = await client.get(
                f"https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json",
                params={"point": f"{lat},{lon}", "key": api_key},
            )
            flow_resp.raise_for_status()
            flow = flow_resp.json().get("flowSegmentData", {})

            current_speed   = float(flow.get("currentSpeed",   50))
            free_flow_speed = float(flow.get("freeFlowSpeed",  80))
            tt_confidence   = float(flow.get("confidence",     0.5))

            speed_ratio   = current_speed / max(free_flow_speed, 1)
            vehicle_count = int((1 - speed_ratio) * 300 + 50)
            prev_state    = (
                0 if speed_ratio > 0.80 else
                1 if speed_ratio > 0.60 else
                2 if speed_ratio > 0.35 else 3
            )

            # Reverse geocode
            rev_resp = await client.get(
                f"https://api.tomtom.com/search/2/reverseGeocode/{lat},{lon}.json",
                params={"key": api_key},
            )
            rev_resp.raise_for_status()
            addresses = rev_resp.json().get("addresses", [])
            road_name = "Unknown Road"
            if addresses:
                addr      = addresses[0].get("address", {})
                road_name = (
                    addr.get("streetName") or
                    addr.get("municipalitySubdivision") or
                    addr.get("municipality") or
                    "Unknown Road"
                )

        road_id = abs(hash(f"{lat:.4f}{lon:.4f}")) % 100000

        return {
            "road_id":           road_id,
            "road_name":         road_name,
            "lat":               lat,
            "lon":               lon,
            "avg_speed":         current_speed,
            "vehicle_count":     vehicle_count,
            "weather":           weather_label,
            "hour":              hour,
            "day_of_week":       now.weekday(),
            "previous_state":    prev_state,
            "accident":          False,
            "free_flow_speed":   free_flow_speed,
            "tomtom_confidence": tt_confidence,
            "data_source":       "tomtom_live",
        }

    except Exception as exc:
        # TomTom failed despite having a key → synthetic fallback
        result = _synthetic_data(lat, lon, hour)
        result["weather"]     = weather_label
        result["data_source"] = f"simulated (tomtom error: {exc})"
        return result
