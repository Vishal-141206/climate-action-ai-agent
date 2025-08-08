# app.py
import streamlit as st
import pandas as pd
import requests
from datetime import datetime
from groq import Groq
import plotly.express as px
import re


# --- PAGE CONFIGURATION & STYLING ---
st.set_page_config(page_title="Climate Action Agent", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Sans+Pro:wght@400;600;700&display=swap');
html, body, [class*="st-"], [class*="css-"] {
    font-family: 'Source Sans Pro', sans-serif;
}
h3 {
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)


# --- CONFIGURATION ---
import os

WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


CO2_DATA_URL = "https://raw.githubusercontent.com/owid/co2-data/master/owid-co2-data.csv"


# --- HELPERS ---
def normalize_headings(md: str) -> str:
    # Remove leading indentation so Markdown headings render
    lines = [line.lstrip() for line in md.splitlines()]
    md = "\n".join(lines)
    # Convert **## Title** or __## Title__ -> ## Title
    md = re.sub(r'^\s*(\*\*|__)\s*(#{1,6}\s+.*?)\s*(\*\*|__)\s*$', r'\2', md, flags=re.MULTILINE)
    # Convert ##** Title ** -> ## Title
    md = re.sub(r'^(#{1,6})\s*[*_]{2}\s*(.+?)\s*[*_]{2}\s*$', r'\1 \2', md, flags=re.MULTILINE)
    return md


# --- AGENT'S CORE FUNCTIONS (with Caching) ---
@st.cache_data
def get_co2_data():
    try:
        df = pd.read_csv(CO2_DATA_URL)
        global_df = df[df['country'] == 'World'][['year', 'co2']].dropna()
        return global_df[global_df['year'] >= 1950]
    except Exception:
        return None


@st.cache_data
def get_groq_prediction(history_df_string, future_year):
    if not GROQ_API_KEY or "your_groq_api_key_here" in GROQ_API_KEY:
        return None
    client = Groq(api_key=GROQ_API_KEY)
    prompt = (
        "Based on this historical CO2 emissions data:\n"
        f"{history_df_string}\n\n"
        f"Predict the value for the year {future_year}. Respond with only the numerical value."
    )
    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-8b-8192"
        )
        return float(chat_completion.choices[0].message.content.strip())
    except Exception:
        return None


@st.cache_data
def get_groq_action_plan(year, predicted_co2, last_known_year, last_known_co2):
    if not GROQ_API_KEY or "your_groq_api_key_here" in GROQ_API_KEY:
        return "## Groq analysis not available."
    client = Groq(api_key=GROQ_API_KEY)

    
    prompt = f"""
You are an expert climate policy strategist and you have to predict and mitigate the impact of climate change by analyzing
historical climate data, current environmental conditions, and human activities. The solution should
help policymakers and organizations develop effective climate action plans. The last recorded global CO2 emission was {last_known_co2:.2f} Mt in {last_known_year}. An AI model predicts that CO2 emissions will reach {predicted_co2:.2f} Mt in the year {year}.
Based on this data, provide a structured action plan with exactly two sections using Markdown subheadings as shown below. Do not add bold around the hashes, do not indent headings, and do not include extra sections.

## Impact of Climate Change
Provide a one-paragraph analysis of the forecast's significance and climate change impact on humans and the Earth.

## Solution to Develop Effective Climate Action Plans
Provide a list of 5 high-impact policy solutions to help policymakers and organizations develop effective climate action plans. Give solution explaining all the imapcts of it on climate change.
""".strip()

    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-8b-8192"
        )
        content = chat_completion.choices[0].message.content
        return normalize_headings(content)
    except Exception as e:
        return f"## Error\nAn error occurred with the Groq API: {e}"


@st.cache_data
def get_groq_follow_up(initial_analysis, user_question):
    if not GROQ_API_KEY or "your_groq_api_key_here" in GROQ_API_KEY:
        return "**Groq is not available.**"
    client = Groq(api_key=GROQ_API_KEY)
    prompt = f"Initial analysis:\n---\n{initial_analysis}\n---\nAnswer this follow-up question: \"{user_question}\""
    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-8b-8192"
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"An error occurred with the Groq API: {e}"


@st.cache_data
def get_weather_and_aqi(city):
    if not WEATHER_API_KEY or "your_weatherapi_key_here" in WEATHER_API_KEY:
        return {"error": "Weather API key not configured."}
    url = f"http://api.weatherapi.com/v1/forecast.json?key={WEATHER_API_KEY}&q={city}&days=1&aqi=yes&alerts=no"
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        return {"error": None, "data": response.json()}
    except Exception:
        return {"error": f"Could not fetch data for '{city}'."}


# --- UI ---
st.title("Climate Action AI Agent")

co2_df = get_co2_data()

if co2_df is not None:
    st.header("Global Climate: Prediction & Mitigation")

    # --- Main Prediction Section ---
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Forecast Future CO₂ Emissions")
        future_year = st.slider(
            "Select a year to predict:",
            min_value=datetime.now().year, max_value=2050, value=2035, step=1
        )
        history_snippet = co2_df.tail(20).to_string(index=False)
        predicted_co2 = get_groq_prediction(history_snippet, future_year)

        st.markdown("##### Historical & Forecasted CO₂ Levels")
        fig = px.line(
            co2_df, x='year', y='co2',
            labels={'year': 'Year', 'co2': 'CO₂ Emissions (Megatonnes)'}
        )
        fig.update_traces(line_color='#FF4B4B', line_width=3, name='Historical')
        if predicted_co2 is not None:
            fig.add_scatter(
                x=[future_year], y=[predicted_co2],
                mode='markers', name='Forecast',
                marker=dict(color='orange', size=12)
            )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Forecast Details")
        if predicted_co2 is not None:
            last_known_co2 = co2_df['co2'].iloc[-1]
            last_known_year = co2_df['year'].iloc[-1]
            change = predicted_co2 - last_known_co2
            st.metric(
                label=f"Predicted CO₂ emission for {future_year}",
                value=f"{predicted_co2:.2f} Mt",
                delta=f"{change:.2f} Mt vs. {last_known_year}",
                delta_color="inverse"
            )
        else:
            st.error("Could not generate a forecast.")

    st.divider()

    # --- Agent's Briefing ---
    st.header("🤖 Agent's Prediction:")
    if predicted_co2 is not None:
        last_known_co2 = co2_df['co2'].iloc[-1]
        last_known_year = co2_df['year'].iloc[-1]
        action_plan = get_groq_action_plan(future_year, predicted_co2, last_known_year, last_known_co2)
        st.session_state['initial_analysis'] = action_plan
        # Render with proper headings
        st.markdown(action_plan)

        st.subheader("May I help you further?")
        follow_up_question = st.text_input("Ask a question about the agent's prediction:", key="follow_up_q")
        if follow_up_question:
            with st.spinner("Agent is thinking..."):
                follow_up_answer = get_groq_follow_up(st.session_state['initial_analysis'], follow_up_question)
                st.info(follow_up_answer)

    st.divider()

    # --- Current Conditions ---
    st.header("🌦️ Current Regional Conditions")
    city_input = st.text_input("Enter a City to Analyze Current Conditions:", "Delhi")
    if city_input:
        weather_data = get_weather_and_aqi(city_input)
        if weather_data['error']:
            st.warning(weather_data['error'])
        else:
            current_data = weather_data['data']['current']
            location = weather_data['data']['location']

            # Display metrics in columns
            metric_col1, metric_col2 = st.columns(2)
            with metric_col1:
                st.metric(label=f"Temperature in {location['name']}", value=f"{current_data['temp_c']} °C")
            with metric_col2:
                aqi_meaning = {
                    1: "Good",
                    2: "Moderate",
                    3: "Unhealthy (Sensitive)",
                    4: "Unhealthy",
                    5: "Very Unhealthy ",
                    6: "Hazardous"
                }
                aqi_index = current_data['air_quality']['us-epa-index']
                st.metric(label="Air Quality (US EPA Index)", value=aqi_meaning.get(aqi_index, "Unknown"))

            #checkbox
            show_details = st.checkbox("Show more weather & pollutant details", value=False)
            if show_details:
                st.write(f"**Condition:** {current_data['condition']['text']}")
                st.write(f"**Wind:** {current_data['wind_kph']} kph")
                st.write(f"**Humidity:** {current_data['humidity']}%")
                st.write(f"**Carbon Monoxide (CO):** {current_data['air_quality']['co']:.2f} μg/m³")
                st.write(f"**Ozone (O₃):** {current_data['air_quality']['o3']:.2f} μg/m³")

else:
    st.error("Could not load historical CO2 data.")
