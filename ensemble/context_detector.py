"""
Traffic Context Detector

Detects the dominant operational context from real-time inputs.
Priority order (highest first):
  1. Accident event
  2. Weather event  (fog / rain / snow)
  3. Morning rush   07:00 – 10:00
  4. Evening rush   17:00 – 20:00
  5. Night          00:00 – 05:00
  6. Normal conditions (default)
"""

from enum import Enum


class TrafficContext(str, Enum):
    MORNING_RUSH      = "morning_rush"
    EVENING_RUSH      = "evening_rush"
    NIGHT_LOW_TRAFFIC = "night_low_traffic"
    WEATHER_EVENT     = "weather_event"
    ACCIDENT_EVENT    = "accident_event"
    NORMAL_CONDITIONS = "normal_conditions"


ADVERSE_WEATHER = frozenset({"rain", "snow", "fog"})


def detect_context(
    hour: int,
    weather: str,
    accident: bool = False,
) -> TrafficContext:
    """Return the most relevant TrafficContext for the given inputs."""
    if accident:
        return TrafficContext.ACCIDENT_EVENT

    if weather.lower() in ADVERSE_WEATHER:
        return TrafficContext.WEATHER_EVENT

    if 7 <= hour <= 10:
        return TrafficContext.MORNING_RUSH

    if 17 <= hour <= 20:
        return TrafficContext.EVENING_RUSH

    if 0 <= hour <= 5:
        return TrafficContext.NIGHT_LOW_TRAFFIC

    return TrafficContext.NORMAL_CONDITIONS
