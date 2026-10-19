"""Lightweight rule-based natural-language entity extraction for bike demand inputs.

This is an educational NLP component using token patterns, aliases and regular
expressions. It does not use a trained language model.
"""
from __future__ import annotations
import re
from typing import Any

SEASONS = {
    "spring": (1, "Spring"), "summer": (2, "Summer"),
    "fall": (3, "Fall"), "autumn": (3, "Fall"),
    "winter": (4, "Winter"),
}
MONTHS = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
    "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6,
    "july": 7, "jul": 7, "august": 8, "aug": 8, "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12,
}
WEEKDAYS = {
    "sunday": 0, "monday": 1, "tuesday": 2, "wednesday": 3,
    "thursday": 4, "friday": 5, "saturday": 6,
}
WEATHER = [
    (r"\b(heavy rain|heavy rainfall|thunderstorm|heavy snow|stormy)\b", 4, "Heavy rain / snow"),
    (r"\b(light rain|drizzle|rainy|snow|snowy)\b", 3, "Light rain / snow"),
    (r"\b(mist|fog|cloudy|overcast)\b", 2, "Mist / cloudy"),
    (r"\b(clear|sunny|fair weather|blue sky)\b", 1, "Clear / partly cloudy"),
]


def _number(pattern: str, text: str) -> float | None:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    return float(match.group(1)) if match else None


def parse_bike_request(text: str) -> dict[str, Any]:
    """Extract supported bike rental conditions from a plain-English request."""
    original = (text or "").strip()
    t = original.lower()
    fields: dict[str, Any] = {}
    found: list[str] = []
    notes: list[str] = []

    # Hour: 8 AM, 8:30 pm (minutes are intentionally ignored for hourly data), 14:00.
    hour_match = re.search(r"\b(\d{1,2})(?::\d{2})?\s*(am|pm)\b", t)
    if hour_match:
        h = int(hour_match.group(1)) % 12
        fields["hour"] = h + (12 if hour_match.group(2) == "pm" else 0)
        found.append("hour")
    else:
        hour_match = re.search(r"\b(?:at|around|hour)\s+(\d{1,2})(?::\d{2})?\b", t)
        if hour_match and 0 <= int(hour_match.group(1)) <= 23:
            fields["hour"] = int(hour_match.group(1)); found.append("hour")

    # Temperature and feels-like temperature.
    feels = _number(r"(?:feels?[- ]like|apparent temperature)\s*(?:is|of|=|:)?\s*(-?\d+(?:\.\d+)?)\s*(?:°?\s*c|celsius)\b", t)
    temp = _number(r"(?<!feels[- ])(?<!feels )\b(?:temperature|temp|at)\s*(?:is|of|=|:)?\s*(-?\d+(?:\.\d+)?)\s*°?\s*(?:c|celsius)\b", t)
    if temp is None:
        temp = _number(r"\b(-?\d+(?:\.\d+)?)\s*°?\s*(?:c|celsius)\b", t)
    if temp is not None:
        if -8 <= temp <= 39:
            fields["temperature"] = temp; found.append("temperature")
        else:
            notes.append("Temperature is outside the app's supported range (-8 to 39 °C).")
    if feels is not None:
        if -16 <= feels <= 50:
            fields["feels_like"] = feels; found.append("feels-like temperature")
        else:
            notes.append("Feels-like temperature is outside the app's supported range (-16 to 50 °C).")

    humidity = _number(r"\b(?:humidity|humid)\s*(?:is|of|=|:)?\s*(\d+(?:\.\d+)?)\s*%?", t)
    if humidity is None:
        humidity = _number(r"\b(\d+(?:\.\d+)?)\s*%\s*(?:humidity|humid)\b", t)
    if humidity is not None:
        if 0 <= humidity <= 100:
            fields["humidity"] = humidity; found.append("humidity")
        else:
            notes.append("Humidity must be between 0 and 100%.")

    wind = _number(r"\b(?:wind(?: speed)?|windspeed)\s*(?:is|of|=|:)?\s*(\d+(?:\.\d+)?)\s*(?:km/?h|kph|km per hour)\b", t)
    if wind is not None:
        if 0 <= wind <= 100:
            fields["wind"] = wind; found.append("wind speed")
        else:
            notes.append("Wind speed is outside the app's supported range (0 to 100 km/h).")

    for name, (sid, label) in SEASONS.items():
        if re.search(rf"\b{re.escape(name)}\b", t):
            fields["season_label"] = label; found.append("season"); break

    for name, month_num in MONTHS.items():
        if re.search(rf"\b{re.escape(name)}\b", t):
            fields["month"] = month_num; found.append("month"); break

    for name, day_num in WEEKDAYS.items():
        if re.search(rf"\b{re.escape(name)}\b", t):
            fields["weekday_label"] = name.title(); found.append("weekday"); break

    if re.search(r"\b(weekday|weekdays|working day|workingday|workday)\b", t):
        fields["workingday_label"] = "Yes"; found.append("working-day status")
    elif re.search(r"\b(weekend|weekends)\b", t):
        fields["workingday_label"] = "No"; found.append("working-day status")

    if re.search(r"\b(holiday|public holiday)\b", t):
        fields["holiday_label"] = "Yes"; found.append("holiday status")
    elif re.search(r"\b(non[- ]holiday|not a holiday|regular day)\b", t):
        fields["holiday_label"] = "No"; found.append("holiday status")

    for pattern, wid, label in WEATHER:
        if re.search(pattern, t):
            fields["weather_label"] = label; found.append("weather"); break

    year_match = re.search(r"\b(2011|2012)\b", t)
    if year_match:
        fields["year_label"] = "2012 (1)" if year_match.group(1) == "2012" else "2011 (0)"
        found.append("dataset year")

    # Season-to-month fallback only when a season is stated but no month is given.
    if "season_label" in fields and "month" not in fields:
        fields["month"] = {"Spring": 4, "Summer": 7, "Fall": 10, "Winter": 1}[fields["season_label"]]
        notes.append("Month was set to a representative month for the stated season; review it before predicting.")

    # For generic weekday/weekend requests, keep the selected weekday unless one was named.
    if "workingday_label" in fields and "weekday_label" not in fields:
        notes.append("A weekday/weekend was identified, but no exact day was given. Review the weekday field.")

    return {
        "original_text": original,
        "fields": fields,
        "entities_found": found,
        "notes": notes,
        "message": "Extracted values are suggestions. Review them before running the prediction.",
        "method": "rule_based_nlp_entity_extraction",
    }
