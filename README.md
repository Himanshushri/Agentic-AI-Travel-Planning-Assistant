# ✈️ Agentic AI-Based Travel Planning Assistant Using LangChain

An autonomous multi-step AI travel planning system built with **Python**, **LangChain**, **Groq / OpenAI LLMs**, **Open-Meteo Live Weather API**, and **Streamlit**.

---

## 📌 Problem Statement
Planning a trip typically forces travelers to switch across multiple websites for flights, hotels, attractions, and weather forecasts while manually calculating budgets. This Agentic AI system automates the entire workflow by reasoning over structured datasets and live weather APIs to produce optimized, justified, day-by-day travel itineraries.

---

## ✨ Key Features
- **Autonomous LangChain Tool-Calling Agent:** Uses native `llm.bind_tools()` with a multi-step `ToolMessage` execution loop across 5 specialized `@tool` functions.
- **Live Real-Time Weather Integration:** Fetches up-to-date daily temperatures and weather conditions via the free **Open-Meteo REST API** (no API key required).
- **Dual-Mode Execution:** Supports **Groq (`llama-3.3-70b-versatile`)**, **OpenAI (`gpt-4o-mini`)**, and a **Built-in ReAct Engine** fallback for uninterrupted reliability.
- **Structured & Human-Readable Outputs:** Displays a clean formatted itinerary alongside downloadable JSON and a full step-by-step **Agent Tool Execution Trace**.
- **Decision Justification:** Explains *why* specific flights, hotels, and attractions were chosen based on price, flight duration, star rating, and user preferences.

---

## 🗂️ Project Architecture

```text
agentic_travel_planner/
├── app.py                     # Streamlit frontend UI (sidebar filters, prompt editor, 3 output tabs)
├── agent/
│   ├── __init__.py            # Package initializer
│   └── planner_agent.py       # LangChain bind_tools() agent loop + built-in ReAct fallback
├── tools/
│   ├── __init__.py            # Package initializer
│   └── travel_tools.py        # 5 custom LangChain @tool functions + Open-Meteo API integration
├── data/
│   ├── flights.json           # 30 flight routes with ISO timestamps and INR fares
│   ├── hotels.json            # 40 hotels across 8 Indian cities with star ratings & amenities
│   └── places.json            # 40 tourist attractions categorized by type and rating
├── requirements.txt           # Python dependencies
├── .env.example               # Template for API key configuration
├── .gitignore                 # Excludes secrets and virtual environments
└── README.md                  # Project documentation
```

---

## 🛠️ Tech Stack
- **Language:** Python 3.10+
- **Agent Framework:** LangChain (`langchain-core`, `langchain-groq`, `langchain-openai`)
- **Weather API:** Open-Meteo Forecast API (Free, Real-Time)
- **Frontend:** Streamlit
- **Environment Management:** `python-dotenv`

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/Himanshushri/Agentic-AI-Travel-Planning-Assistant.git
cd Agentic-AI-Travel-Planning-Assistant
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure API Key (Optional)
Create a `.env` file in the root directory (see `.env.example`) or enter your Groq API key directly in the Streamlit sidebar:
```env
GROQ_API_KEY=your_groq_api_key_here
```

### 4. Run the Application
```bash
streamlit run app.py
```

---

## 👤 Author
**Himanshu Shrivastava**  
AI/ML Intern — Labmentix
