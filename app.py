"""
Streamlit Frontend for the Agentic AI-Based Travel Planning Assistant.
Features interactive sidebar preferences, editable natural-language prompt box,
and tabbed output (Human-Readable Itinerary, Structured JSON, and Agent Tool Trace).
"""

import json
import os
import streamlit as st
from dotenv import load_dotenv

from tools.travel_tools import _load_json
from agent.planner_agent import run_autonomous_react_simulation, run_live_llm_agent

load_dotenv()


def _get_secret(key_name: str) -> str:
    """Safely fetches an API key from .env or Streamlit Cloud Secrets without throwing local errors."""
    val = os.getenv(key_name, "")
    if val:
        return val
    try:
        return str(st.secrets.get(key_name, ""))
    except Exception:
        return ""


st.set_page_config(
    page_title="Agentic AI Travel Planner",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("✈️ Agentic AI-Based Travel Planning Assistant")
st.markdown(
    "Powered by **LangChain Tool-Calling Agents** & **Open-Meteo Live Weather API** | *Built by Himanshu Shrivastava*"
)

cities = ["Delhi", "Mumbai", "Goa", "Bangalore", "Chennai", "Hyderabad", "Kolkata", "Jaipur"]
place_types = ["All", "beach", "fort", "lake", "market", "monument", "museum", "park", "temple"]

# Sidebar for LLM Configuration & Trip Preferences
with st.sidebar:
    st.header("⚙️ LLM & Agent Settings")
    llm_provider = st.selectbox(
        "Select LLM Provider",
        options=[
            "Groq (Llama-3.3-70B)",
            "OpenAI (GPT-4o-mini)",
            "Built-in ReAct Engine (No API Key)"
        ]
    )

    provider_key = "groq" if "Groq" in llm_provider else ("openai" if "OpenAI" in llm_provider else "builtin")

    api_key = ""
    if provider_key == "groq":
        api_key = st.text_input(
            "Enter GROQ API Key",
            value=_get_secret("GROQ_API_KEY"),
            type="password",
            help="Paste your Groq API key (gsk_...) or configure GROQ_API_KEY in Streamlit Secrets."
        )
    elif provider_key == "openai":
        api_key = st.text_input(
            "Enter OPENAI API Key",
            value=_get_secret("OPENAI_API_KEY"),
            type="password",
            help="Paste your OpenAI API key (sk-...) or configure OPENAI_API_KEY in Streamlit Secrets."
        )

    st.divider()
    st.header("🗺️ Trip Preferences")
    source_city = st.selectbox("Source City", options=cities, index=4)       # Default: Chennai
    destination_city = st.selectbox("Destination City", options=cities, index=1) # Default: Mumbai
    num_days = st.slider("Trip Duration (Days)", min_value=2, max_value=7, value=3)
    flight_pref = st.radio("Flight Optimization", options=["Cheapest", "Fastest"], horizontal=True)
    max_hotel_budget = st.number_input(
        "Max Hotel Price / Night (₹)", min_value=1000, max_value=15000, value=4500, step=500
    )
    min_hotel_rating = st.slider(
        "Minimum Hotel Star Rating", min_value=2, max_value=5, value=3, step=1
    )
    interest_type = st.selectbox("Primary Interest", options=place_types, index=0)

# Dataset Overview Expander
with st.expander("📊 View Dataset Overview (`flights.json`, `hotels.json`, `places.json`)"):
    col1, col2, col3 = st.columns(3)
    flights_raw = _load_json("flights.json")
    hotels_raw = _load_json("hotels.json")
    places_raw = _load_json("places.json")
    with col1:
        st.metric("Flights Loaded", len(flights_raw))
        routes = sorted(list({f"{f['from']} ➔ {f['to']}" for f in flights_raw}))
        st.caption("Sample Direct Routes: " + ", ".join(routes[:6]) + "...")
    with col2:
        st.metric("Hotels Loaded", len(hotels_raw))
        st.caption("8 Major Indian Cities Covered")
    with col3:
        st.metric("Attractions Loaded", len(places_raw))
        st.caption("8 Categories: Beach, Fort, Temple, Museum, etc.")

# Dynamic Editable Prompt Box
constructed_query = (
    f"Plan a {num_days}-day trip from {source_city} to {destination_city}. "
    f"I prefer the {flight_pref.lower()} flight. "
    f"For accommodation, find a hotel under ₹{max_hotel_budget} per night with at least {min_hotel_rating} stars. "
    f"My preferred attraction category is '{interest_type}'. "
    f"Include real-time weather forecasts for {num_days} days, a complete budget breakdown for {num_days - 1} nights, "
    f"and explain why you selected these options."
)

st.subheader("📝 Your Travel Prompt")
custom_query = st.text_area(
    "Edit the travel request below or adjust the sidebar controls:",
    value=constructed_query,
    height=95
)

if st.button("🚀 Generate Autonomous Trip Plan", type="primary", use_container_width=True):
    with st.spinner("🤖 Agent is reasoning, calling LangChain tools, and fetching live Open-Meteo weather..."):
        try:
            if provider_key in ("groq", "openai") and api_key.strip():
                result = run_live_llm_agent(
                    provider=provider_key,
                    api_key=api_key.strip(),
                    user_query=custom_query
                )
            else:
                result = run_autonomous_react_simulation(
                    source=source_city,
                    destination=destination_city,
                    num_days=num_days,
                    flight_pref=flight_pref,
                    max_hotel_budget=max_hotel_budget,
                    min_stars=min_hotel_rating,
                    interest_type=interest_type
                )
        except Exception as exc:
            st.warning(f"Live LLM API returned an error ({exc}). Seamlessly switched to Built-in ReAct Engine.")
            result = run_autonomous_react_simulation(
                source=source_city,
                destination=destination_city,
                num_days=num_days,
                flight_pref=flight_pref,
                max_hotel_budget=max_hotel_budget,
                min_stars=min_hotel_rating,
                interest_type=interest_type
            )

        tab1, tab2, tab3 = st.tabs([
            "📅 Human-Readable Itinerary",
            "🧾 Clean JSON Output",
            "🧠 Agent Reasoning & Tool Logs"
        ])

        with tab1:
            st.markdown(result["readable"])

        with tab2:
            st.subheader("Structured JSON Output")
            st.json(result["json_data"])
            st.download_button(
                label="⬇️ Download Itinerary JSON",
                data=json.dumps(result["json_data"], indent=2),
                file_name=f"{source_city.lower()}_to_{destination_city.lower()}_itinerary.json",
                mime="application/json"
            )

        with tab3:
            st.subheader("Multi-Step Agent Tool Execution Trace")
            if result["steps"]:
                for idx, (tool_name, tool_in, tool_out) in enumerate(result["steps"], start=1):
                    with st.expander(f"Step {idx}: Called Tool `{tool_name}`", expanded=True):
                        st.markdown("**Tool Input Arguments:**")
                        st.json(tool_in)
                        st.markdown("**Tool Output (Observation):**")
                        st.json(tool_out)
            else:
                st.info("Agent completed response directly.")
