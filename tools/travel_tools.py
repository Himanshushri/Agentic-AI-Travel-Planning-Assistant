"""
LangChain Custom Tools for the Agentic AI-Based Travel Planning Assistant.
Tailored to flights.json, hotels.json, places.json, and live Open-Meteo API.
"""

import json
import os
from datetime import datetime
from typing import Dict, Any, List
import requests
from langchain_core.tools import tool

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

CITY_COORDINATES = {
    "goa": (15.2993, 74.1240),
    "delhi": (28.6139, 77.2090),
    "mumbai": (19.0760, 72.8777),
    "jaipur": (26.9124, 75.7873),
    "bangalore": (12.9716, 77.5946),
    "chennai": (13.0827, 80.2707),
    "kolkata": (22.5726, 88.3639),
    "hyderabad": (17.3850, 78.4867),
}

WEATHER_CODES = {
    0: "Sunny / Clear Sky",
    1: "Mainly Clear",
    2: "Partly Cloudy",
    3: "Overcast",
    45: "Foggy",
    51: "Light Drizzle",
    61: "Light Rain",
    63: "Moderate Rain",
    65: "Heavy Rain",
    80: "Rain Showers",
    95: "Thunderstorm",
}


def _load_json(filename: str) -> List[Dict[str, Any]]:
    """Safely loads a JSON dataset from the project's data/ directory."""
    filepath = os.path.join(DATA_DIR, filename)
    with open(filepath, "r", encoding="utf-8") as file:
        return json.load(file)


def _compute_duration_hours(dep_str: str, arr_str: str) -> float:
    """Computes flight duration in hours from ISO 8601 departure and arrival timestamps."""
    try:
        dep = datetime.fromisoformat(dep_str)
        arr = datetime.fromisoformat(arr_str)
        return round((arr - dep).total_seconds() / 3600.0, 2)
    except Exception:
        return 2.5


@tool
def search_flights(source: str, destination: str, sort_by: str = "cheapest") -> str:
    """
    Searches flights.json for flights matching 'from' (source) and 'to' (destination).
    Args:
        source: Departure city (e.g., 'Chennai', 'Delhi', 'Mumbai', 'Bangalore').
        destination: Arrival city (e.g., 'Mumbai', 'Goa', 'Kolkata', 'Jaipur').
        sort_by: Either 'cheapest' or 'fastest'.
    """
    flights = _load_json("flights.json")
    src = source.strip().lower()
    dst = destination.strip().lower()

    matched = []
    for f in flights:
        if f.get("from", "").lower() == src and f.get("to", "").lower() == dst:
            item = dict(f)
            item["duration_hours"] = _compute_duration_hours(
                f.get("departure_time", ""), f.get("arrival_time", "")
            )
            matched.append(item)

    if not matched:
        from_city = [
            dict(f, duration_hours=_compute_duration_hours(f.get("departure_time", ""), f.get("arrival_time", "")))
            for f in flights if f.get("from", "").lower() == src
        ]
        to_city = [
            dict(f, duration_hours=_compute_duration_hours(f.get("departure_time", ""), f.get("arrival_time", "")))
            for f in flights if f.get("to", "").lower() == dst
        ]
        return json.dumps({
            "status": "no_direct_flight_in_dataset",
            "note": f"No direct flight from {source} to {destination} in flights.json. Returning closest available regional options.",
            "flights_to_destination": to_city[:3],
            "flights_from_source": from_city[:3]
        }, indent=2)

    if sort_by.lower() == "fastest":
        matched.sort(key=lambda x: (x["duration_hours"], x["price"]))
    else:
        matched.sort(key=lambda x: (x["price"], x["duration_hours"]))

    return json.dumps({
        "status": "success",
        "criteria": sort_by,
        "recommended_flight": matched[0],
        "alternatives": matched[1:3]
    }, indent=2)


@tool
def recommend_hotels(city: str, max_price_per_night: int = 7000, min_stars: int = 3) -> str:
    """
    Searches hotels.json and filters hotels by city, price_per_night (INR), and star rating.
    Args:
        city: Destination city name.
        max_price_per_night: Maximum budget per night in INR.
        min_stars: Minimum star rating (2 to 5).
    """
    hotels = _load_json("hotels.json")
    city_clean = city.strip().lower()

    city_hotels = [h for h in hotels if h.get("city", "").lower() == city_clean]
    filtered = [
        h for h in city_hotels
        if h.get("price_per_night", 99999) <= int(max_price_per_night)
        and h.get("stars", 0) >= int(min_stars)
    ]

    if not filtered:
        filtered = city_hotels

    filtered.sort(key=lambda x: (-x.get("stars", 0), x.get("price_per_night", 99999)))

    return json.dumps({
        "status": "success",
        "city": city,
        "recommended_hotel": filtered[0] if filtered else None,
        "alternatives": filtered[1:3]
    }, indent=2)


@tool
def discover_places(city: str, preferred_type: str = "all") -> str:
    """
    Searches places.json to recommend top-rated tourist attractions in a city.
    Args:
        city: Destination city name.
        preferred_type: Category filter ('all', 'lake', 'temple', 'museum', 'park', 'fort', 'beach', 'market', 'monument').
    """
    places = _load_json("places.json")
    city_clean = city.strip().lower()
    type_clean = preferred_type.strip().lower()

    city_places = [p for p in places if p.get("city", "").lower() == city_clean]

    if type_clean not in ("all", ""):
        typed = [p for p in city_places if p.get("type", "").lower() == type_clean]
        if typed:
            city_places = typed

    city_places.sort(key=lambda x: -float(x.get("rating", 0)))

    return json.dumps({
        "status": "success",
        "city": city,
        "attractions": city_places
    }, indent=2)


@tool
def get_weather_forecast(city: str, num_days: int = 3) -> str:
    """
    Calls the live Open-Meteo REST API (free, no API key required) to fetch daily weather forecasts.
    Args:
        city: Destination city name.
        num_days: Number of trip days (1 to 7).
    """
    lat, lon = CITY_COORDINATES.get(city.strip().lower(), (15.2993, 74.1240))
    try:
        url = (
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            f"&daily=temperature_2m_max,temperature_2m_min,weathercode"
            f"&timezone=auto"
        )
        resp = requests.get(url, timeout=8)
        resp.raise_for_status()
        daily = resp.json().get("daily", {})
        dates = daily.get("time", [])
        max_t = daily.get("temperature_2m_max", [])
        min_t = daily.get("temperature_2m_min", [])
        codes = daily.get("weathercode", [])

        forecasts = []
        for i in range(min(int(num_days), len(dates))):
            cond = WEATHER_CODES.get(codes[i], "Pleasant")
            forecasts.append({
                "day": f"Day {i + 1}",
                "date": dates[i],
                "condition": cond,
                "max_temp_c": max_t[i],
                "min_temp_c": min_t[i],
                "summary": f"{cond} ({max_t[i]}°C)"
            })
        return json.dumps({"status": "success", "city": city, "forecast": forecasts}, indent=2)
    except Exception:
        return json.dumps({
            "status": "fallback_weather",
            "city": city,
            "forecast": [
                {"day": f"Day {i + 1}", "condition": "Sunny", "max_temp_c": 30 + (i % 3), "summary": f"Sunny ({30 + (i % 3)}°C)"}
                for i in range(int(num_days))
            ]
        }, indent=2)


@tool
def calculate_trip_budget(
    flight_cost: float,
    hotel_price_per_night: float,
    num_nights: int,
    daily_local_expense: float = 1000.0
) -> str:
    """
    Calculates total trip budget in INR by summing flight, hotel nights, and daily food/local travel.
    Args:
        flight_cost: Selected flight cost in INR.
        hotel_price_per_night: Selected hotel price per night in INR.
        num_nights: Number of nights stayed.
        daily_local_expense: Estimated daily food and local commute cost in INR.
    """
    hotel_total = float(hotel_price_per_night) * int(num_nights)
    local_total = float(daily_local_expense) * (int(num_nights) + 1)
    grand_total = float(flight_cost) + hotel_total + local_total

    return json.dumps({
        "flight_cost": round(float(flight_cost)),
        "hotel_cost": round(hotel_total),
        "food_and_local_travel": round(local_total),
        "total_cost": round(grand_total)
    }, indent=2)


def get_all_travel_tools() -> List:
    """Returns all 5 registered LangChain @tool functions."""
    return [
        search_flights,
        recommend_hotels,
        discover_places,
        get_weather_forecast,
        calculate_trip_budget
    ]
