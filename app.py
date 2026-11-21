"""
MyTrip AI — Unified Dark-Mode Conversational Travel Concierge (Streamlit).
Features a single cohesive theme, zero sidebar clutter, optional inline filter toggle,
tap-based quick-start chips, rich visual cards, and silent LLM/ReAct failover.
"""

import json
import os
import re
import streamlit as st
from dotenv import load_dotenv

from tools.travel_tools import _load_json
from agent.planner_agent import run_autonomous_react_simulation, run_live_llm_agent

load_dotenv()

st.set_page_config(
    page_title="MyTrip AI | Agentic Travel Concierge",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Unified Deep-Navy MakeMyTrip-Grade Dark Theme ---
st.markdown("""
<style>
    /* Force unified dark palette across entire app, header, and bottom chat bar */
    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stHeader"],
    [data-testid="stBottom"],
    [data-testid="stBottomBlockContainer"] {
        background-color: #0b1220 !important;
        color: #f8fafc !important;
    }
    /* Hide sidebar completely */
    [data-testid="stSidebar"],
    [data-testid="collapsedControl"] {
        display: none !important;
    }
    /* Ensure all markdown, headers, labels, and tabs are crisp and readable */
    .stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown h1, .stMarkdown h2,
    .stMarkdown h3, .stMarkdown h4, label, .stCaption {
        color: #e2e8f0 !important;
    }
    /* Hero Banner */
    .mmt-hero {
        background: linear-gradient(135deg, #0f1e36 0%, #132c52 55%, #0c4a6e 100%);
        border: 1px solid #1e3a5f;
        padding: 1.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 1.2rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
    }
    .mmt-badge {
        display: inline-block;
        background: rgba(56, 189, 248, 0.15);
        border: 1px solid rgba(56, 189, 248, 0.45);
        color: #7dd3fc;
        padding: 4px 12px;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        margin-bottom: 0.5rem;
    }
    .mmt-hero h1 {
        margin: 0;
        font-size: 1.85rem;
        font-weight: 700;
        color: #ffffff !important;
    }
    .mmt-hero p {
        margin: 0.35rem 0 0 0;
        color: #cbd5e1 !important;
        font-size: 0.95rem;
    }
    /* Unified Dark Elevated Cards */
    .travel-card {
        background: #131d31;
        border-radius: 14px;
        padding: 1.2rem 1.4rem;
        border: 1px solid #24344d;
        box-shadow: 0 6px 18px rgba(0, 0, 0, 0.25);
        margin-bottom: 1rem;
        color: #f8fafc;
    }
    .card-header {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #38bdf8;
        font-weight: 700;
        margin-bottom: 0.35rem;
    }
    .card-title {
        font-size: 1.22rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 0.25rem;
    }
    .price-tag {
        font-size: 1.3rem;
        font-weight: 800;
        color: #38bdf8;
    }
    .pill {
        display: inline-block;
        background: #1e293b;
        border: 1px solid #334155;
        color: #93c5fd;
        padding: 3px 10px;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-right: 6px;
        margin-top: 6px;
    }
    .weather-box {
        background: #131d31;
        border: 1px solid #24344d;
        border-top: 3px solid #38bdf8;
        border-radius: 12px;
        padding: 0.85rem;
        text-align: center;
        color: #f8fafc;
    }
    .day-card {
        background: #131d31;
        border: 1px solid #24344d;
        border-left: 4px solid #38bdf8;
        border-radius: 10px;
        padding: 0.9rem 1.1rem;
        margin-bottom: 0.7rem;
        color: #e2e8f0;
    }
    .budget-total {
        background: linear-gradient(93deg, #0284c7 0%, #2563eb 100%);
        color: #ffffff;
        padding: 0.9rem 1.25rem;
        border-radius: 10px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-weight: 700;
        font-size: 1.1rem;
        margin-top: 0.75rem;
    }
    /* Chat messages unified styling */
    [data-testid="stChatMessage"] {
        background-color: #111a2c !important;
        border: 1px solid #1e2d45;
        border-radius: 14px;
        padding: 1rem;
        margin-bottom: 0.8rem;
    }
    /* Quick-start chip buttons */
    div.stButton > button {
        background-color: #131d31;
        color: #e2e8f0;
        border: 1px solid #2a3c58;
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    div.stButton > button:hover {
        border-color: #38bdf8;
        color: #ffffff;
        background-color: #1a2942;
    }
    div.stButton > button[kind="primary"] {
        background: linear-gradient(93deg, #0284c7 0%, #2563eb 100%);
        border: none;
        color: #ffffff;
        border-radius: 10px;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)


def _get_secret(key_name: str) -> str:
    """Safely fetches an API key from .env or Streamlit Cloud Secrets."""
    val = os.getenv(key_name, "")
    if val:
        return val
    try:
        return str(st.secrets.get(key_name, ""))
    except Exception:
        return ""


def _execute_with_failover(
    user_query: str,
    source: str,
    destination: str,
    num_days: int,
    flight_pref: str,
    max_hotel_budget: int,
    min_stars: int,
    interest_type: str
):
    """Runs Live LLM (Groq/OpenAI) if configured, with silent automatic failover to Built-in ReAct."""
    groq_key = _get_secret("GROQ_API_KEY").strip()
    openai_key = _get_secret("OPENAI_API_KEY").strip()
    provider = "groq" if groq_key else ("openai" if openai_key else "")
    api_key = groq_key or openai_key

    if provider and api_key:
        try:
            res = run_live_llm_agent(provider=provider, api_key=api_key, user_query=user_query)
            if not isinstance(res.get("json_data"), dict) or "flight_selected" not in res["json_data"]:
                sim = run_autonomous_react_simulation(
                    source, destination, num_days, flight_pref, max_hotel_budget, min_stars, interest_type
                )
                res["json_data"] = sim["json_data"]
            return res
        except Exception:
            pass

    return run_autonomous_react_simulation(
        source=source,
        destination=destination,
        num_days=num_days,
        flight_pref=flight_pref,
        max_hotel_budget=max_hotel_budget,
        min_stars=min_stars,
        interest_type=interest_type
    )


def _render_rich_trip_cards(result: dict, source: str, destination: str):
    """Renders high-contrast unified dark cards for Flight, Hotel, Weather, Itinerary, and Budget."""
    data = result.get("json_data", {})
    flight = data.get("flight_selected", {})
    hotel = data.get("hotel_booked", {})
    weather = data.get("weather_forecast", [])
    itinerary = data.get("day_wise_itinerary", [])
    budget = data.get("budget_breakdown", {})
    reasons = data.get("decision_justification", {})

    col_f, col_h = st.columns(2)
    with col_f:
        dep_raw = str(flight.get("departure_time", "14:00"))
        dep_time = dep_raw.split("T")[-1][:5] if "T" in dep_raw else dep_raw
        dep_date = dep_raw[:10] if "T" in dep_raw else "Scheduled"
        st.markdown(f"""
        <div class="travel-card">
            <div class="card-header">✈️ Selected Flight · {flight.get('flight_id', 'FL')}</div>
            <div class="card-title">{flight.get('airline', 'Airline')} &nbsp; <span class="price-tag">₹{int(flight.get('price', 0)):,}</span></div>
            <div style="font-size:0.95rem; color:#cbd5e1; margin-top:6px;">
                <b>{flight.get('from', source)}</b> ➔ <b>{flight.get('to', destination)}</b>
            </div>
            <div style="margin-top:8px;">
                <span class="pill">🕒 Departs {dep_time}</span>
                <span class="pill">📅 {dep_date}</span>
                <span class="pill">⏱️ {flight.get('duration_hours', 2.0)} hrs</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_h:
        stars_count = int(hotel.get("stars", 4))
        stars_str = "⭐" * stars_count
        amenities_html = "".join(
            [f'<span class="pill">{a.title()}</span>' for a in hotel.get("amenities", ["WiFi", "Pool"])]
        )
        st.markdown(f"""
        <div class="travel-card">
            <div class="card-header">🏨 Recommended Stay · {hotel.get('hotel_id', 'HOT')}</div>
            <div class="card-title">{hotel.get('name', 'Hotel')} &nbsp; <span class="price-tag">₹{int(hotel.get('price_per_night', 0)):,}/night</span></div>
            <div style="font-size:0.9rem; color:#cbd5e1; margin-top:4px;">{stars_str} ({stars_count}-Star Property)</div>
            <div style="margin-top:8px;">{amenities_html}</div>
        </div>
        """, unsafe_allow_html=True)

    if weather:
        st.markdown("#### 🌤️ Live Weather Forecast (Open-Meteo API)")
        w_cols = st.columns(min(len(weather), 5))
        for idx, w in enumerate(weather[:5]):
            with w_cols[idx]:
                st.markdown(f"""
                <div class="weather-box">
                    <div style="font-weight:700; font-size:0.82rem; color:#93c5fd;">{w.get('day', f'Day {idx+1}')}</div>
                    <div style="font-size:1.35rem; font-weight:800; margin:4px 0; color:#ffffff;">{w.get('max_temp_c', 30)}°C</div>
                    <div style="font-size:0.78rem; color:#cbd5e1;">{w.get('condition', 'Sunny')}</div>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("")
    col_it, col_bg = st.columns([1.4, 1])
    with col_it:
        st.markdown("#### 🗺️ Day-Wise Smart Itinerary")
        for day in itinerary:
            st.markdown(f"""
            <div class="day-card">
                <div style="font-weight:700; color:#38bdf8; margin-bottom:4px;">{day.get('day', 'Day')}</div>
                <div style="font-size:0.9rem;"><b>🌅 Morning:</b> {day.get('morning', '')}</div>
                <div style="font-size:0.9rem; margin-top:3px;"><b>🌆 Evening:</b> {day.get('evening', '')}</div>
            </div>
            """, unsafe_allow_html=True)

    with col_bg:
        st.markdown("#### 💰 Estimated Trip Budget")
        st.markdown(f"""
        <div class="travel-card">
            <div style="display:flex; justify-content:space-between; margin-bottom:8px; color:#cbd5e1;">
                <span>✈️ Flight Fare</span><b style="color:#ffffff;">₹{int(budget.get('flight_cost', 0)):,}</b>
            </div>
            <div style="display:flex; justify-content:space-between; margin-bottom:8px; color:#cbd5e1;">
                <span>🏨 Hotel Stay</span><b style="color:#ffffff;">₹{int(budget.get('hotel_cost', 0)):,}</b>
            </div>
            <div style="display:flex; justify-content:space-between; margin-bottom:8px; color:#cbd5e1;">
                <span>🍽️ Food & Local Travel</span><b style="color:#ffffff;">₹{int(budget.get('food_and_local_travel', 0)):,}</b>
            </div>
            <div class="budget-total">
                <span>Total Estimate</span>
                <span>₹{int(budget.get('total_cost', 0)):,}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if reasons:
            st.markdown(f"""
            <div class="travel-card" style="border-left: 4px solid #38bdf8;">
                <div class="card-header">🧠 Why We Selected This</div>
                <div style="font-size:0.86rem; color:#cbd5e1; margin-bottom:6px;"><b>• Flight:</b> {reasons.get('flight_reason', '')}</div>
                <div style="font-size:0.86rem; color:#cbd5e1;"><b>• Hotel:</b> {reasons.get('hotel_reason', '')}</div>
            </div>
            """, unsafe_allow_html=True)

    t1, t2, t3 = st.tabs(["📄 Full Written Plan", "🧾 Structured JSON", "🧠 Agent Tool Trace"])
    with t1:
        st.markdown(result.get("readable", ""))
    with t2:
        st.json(data)
        st.download_button(
            label="⬇️ Download Itinerary JSON",
            data=json.dumps(data, indent=2),
            file_name=f"{source.lower()}_to_{destination.lower()}_trip.json",
            mime="application/json"
        )
    with t3:
        for idx, (tool_name, tool_in, tool_out) in enumerate(result.get("steps", []), start=1):
            with st.expander(f"Step {idx}: LangChain Tool `{tool_name}`"):
                st.markdown("**Inputs:**")
                st.json(tool_in)
                st.markdown("**Observation:**")
                st.json(tool_out)


# --- Hero Header ---
st.markdown("""
<div class="mmt-hero">
    <div class="mmt-badge">● AI TRAVEL CONCIERGE · AUTONOMOUS MULTI-TOOL AGENT</div>
    <h1>✈️ MyTrip AI — Smart Travel Planning Assistant</h1>
    <p>Chat naturally below, tap a quick-start trip, or toggle optional filters. Live Open-Meteo weather, optimized flights, curated stays, and instant budgets in one unified interface.</p>
</div>
""", unsafe_allow_html=True)

cities = ["Delhi", "Mumbai", "Goa", "Bangalore", "Chennai", "Hyderabad", "Kolkata", "Jaipur"]
place_types = ["All", "beach", "fort", "lake", "market", "monument", "museum", "park", "temple"]

# --- Optional Inline Filter Toggle (Replaces Sidebar!) ---
filter_generate = False
with st.expander("🎛️ Customize Trip Filters (Optional — or just type in the chat below)", expanded=False):
    c1, c2, c3, c4 = st.columns(4)
    source_city = c1.selectbox("From (Source)", options=cities, index=4)
    destination_city = c2.selectbox("To (Destination)", options=cities, index=1)
    num_days = c3.slider("Duration (Days)", min_value=2, max_value=7, value=3)
    flight_pref = c4.radio("Flight Priority", options=["Cheapest", "Fastest"], horizontal=True)

    c5, c6, c7, c8 = st.columns([1.2, 1.2, 1.2, 1.4])
    max_hotel_budget = c5.number_input("Max Hotel / Night (₹)", min_value=1000, max_value=15000, value=4500, step=500)
    min_hotel_rating = c6.slider("Min Hotel Stars", min_value=2, max_value=5, value=3)
    interest_type = c7.selectbox("Primary Interest", options=place_types, index=0)
    with c8:
        st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
        filter_generate = st.button("🚀 Plan with These Filters", type="primary", use_container_width=True)

# --- Tap-Based Quick-Start Chips ---
chip1, chip2, chip3, chip4 = st.columns(4)
quick_trigger = None

if chip1.button("🏖️ Mumbai ➔ Goa (3 Days)", use_container_width=True):
    quick_trigger = ("Mumbai", "Goa", 3, "Cheapest", 4000, 4, "all", "Plan a 3-day beach & fort trip from Mumbai to Goa with the cheapest flight and a 4-star hotel.")
if chip2.button("🏰 Kolkata ➔ Jaipur (4 Days)", use_container_width=True):
    quick_trigger = ("Kolkata", "Jaipur", 4, "Fastest", 4500, 4, "park", "Plan a 4-day royal trip from Kolkata to Jaipur with the fastest flight and top-rated attractions.")
if chip3.button("🌆 Chennai ➔ Mumbai (3 Days)", use_container_width=True):
    quick_trigger = ("Chennai", "Mumbai", 3, "Cheapest", 5500, 3, "all", "Plan a 3-day trip from Chennai to Mumbai under ₹5,500/night hotel budget.")
if chip4.button("🛕 Bangalore ➔ Delhi (3 Days)", use_container_width=True):
    quick_trigger = ("Bangalore", "Delhi", 3, "Cheapest", 4500, 4, "lake", "Plan a 3-day cultural trip from Bangalore to Delhi with a 4-star hotel.")

# --- Chat History ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🤖" if msg["role"] == "assistant" else "🧑"):
        if msg["role"] == "user":
            st.markdown(msg["content"])
        else:
            _render_rich_trip_cards(msg["result"], msg["source"], msg["destination"])

# --- Integrated Chat Input ---
chat_prompt = st.chat_input("Ask MyTrip AI anything... e.g. 'Plan a 3-day Goa trip from Mumbai under ₹4,000/night'")

active_request = None
if quick_trigger:
    active_request = quick_trigger
elif filter_generate:
    auto_prompt = (
        f"Plan a {num_days}-day trip from {source_city} to {destination_city}. "
        f"Prefer the {flight_pref.lower()} flight, a hotel under ₹{max_hotel_budget}/night with at least {min_hotel_rating} stars, "
        f"and focus on '{interest_type}' attractions."
    )
    active_request = (source_city, destination_city, num_days, flight_pref, max_hotel_budget, min_hotel_rating, interest_type, auto_prompt)
elif chat_prompt:
    detected = [c for c in cities if c.lower() in chat_prompt.lower()]
    src = detected[0] if len(detected) >= 2 else source_city
    dst = detected[1] if len(detected) >= 2 else (detected[0] if len(detected) == 1 else destination_city)
    days_match = re.search(r"(\d+)\s*-?\s*day", chat_prompt.lower())
    parsed_days = min(max(int(days_match.group(1)), 2), 7) if days_match else num_days
    active_request = (src, dst, parsed_days, flight_pref, max_hotel_budget, min_hotel_rating, interest_type, chat_prompt)

if active_request:
    src, dst, days, f_pref, h_budget, h_stars, interest, prompt_text = active_request

    st.session_state.messages.append({"role": "user", "content": prompt_text})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt_text)

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("✨ Searching flights, curating stays, checking live Open-Meteo weather & optimizing budget..."):
            result = _execute_with_failover(
                user_query=prompt_text,
                source=src,
                destination=dst,
                num_days=days,
                flight_pref=f_pref,
                max_hotel_budget=h_budget,
                min_stars=h_stars,
                interest_type=interest
            )
            _render_rich_trip_cards(result, src, dst)

    st.session_state.messages.append({
        "role": "assistant",
        "result": result,
        "source": src,
        "destination": dst
    })