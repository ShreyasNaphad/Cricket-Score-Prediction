import streamlit as st
import pickle
import pandas as pd
import numpy as np
import base64
import xgboost
from PIL import Image

# --- Page and Model Setup ---

# Load the icon image for the page config
try:
    img = Image.open('ICCMT20WC2021_2020-symbol.png')
    st.set_page_config(page_title='Cricket Score Predictor', page_icon=img, layout="wide")
except FileNotFoundError:
    st.set_page_config(page_title='Cricket Score Predictor 🏏', layout="wide")


# Function to get an image as a base64 string
def get_base64_of_bin_file(bin_file):
    try:
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except FileNotFoundError:
        return None


# --- Load Your Model and Data ---
try:
    pipe = pickle.load(open('pipe (4).pkl', 'rb'))
except FileNotFoundError:
    st.error("🚨 Model file ('pipe (4).pkl') not found. Please ensure it's in the correct directory.")
    st.stop()

teams = [
    'Australia', 'India', 'Bangladesh', 'New Zealand', 'South Africa',
    'England', 'West Indies', 'Afghanistan', 'Pakistan', 'Sri Lanka'
]

cities = [
    'Colombo', 'Mirpur', 'Johannesburg', 'Dubai', 'Auckland', 'Cape Town',
    'London', 'Pallekele', 'Barbados', 'Sydney', 'Melbourne', 'Durban',
    'St Lucia', 'Wellington', 'Lauderhill', 'Hamilton', 'Centurion',
    'Manchester', 'Abu Dhabi', 'Mumbai', 'Nottingham', 'Southampton',
    'Mount Maunganui', 'Chittagong', 'Kolkata', 'Lahore', 'Delhi', 'Nagpur',
    'Chandigarh', 'Adelaide', 'Bangalore', 'St Kitts', 'Cardiff',
    'Christchurch', 'Trinidad'
]

# --- Logo and UI Styling (CSS) ---

team_logos = {
    'Australia': 'AUS.png', 'India': 'BCCI.png', 'Bangladesh': 'BAN1.png',
    'New Zealand': 'NZ1.png', 'South Africa': 'RSA.png', 'England': 'England.png',
    'West Indies': 'WI.png', 'Afghanistan': 'AFG.png', 'Pakistan': 'PCB.png',
    'Sri Lanka': 'SL.png'
}
st.markdown(
    f"""<style>
    .stApp {{
        background-image: url("data:image/jpeg;base64,{base64.b64encode(open("new_bgg.jpg", "rb").read()).decode()}");
        background-size: cover;
        background-repeat: no-repeat;
        background-position: center;
        background-attachment: fixed;
    }}
    </style>""",
    unsafe_allow_html=True
)


def load_css_and_background():
    bg_image_base64 = get_base64_of_bin_file("new_bg.jpg")
    if bg_image_base64:
        background_style = f'background-image: url("data:image/jpeg;base64,{bg_image_base64}");'
    else:
        st.warning("`background.jpg` not found! Displaying a plain dark background.")
        background_style = "background-color: #0f0c29;"

    css = f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');
        [data-testid="stAppViewContainer"] > .main {{
            {background_style}
            background-size: cover; background-position: center; background-repeat: no-repeat; background-attachment: fixed;
        }}
        .stApp {{ color: #ffffff; }}
        .title-container h1 {{
            font-family: 'Poppins', sans-serif; font-size: 3.5rem; font-weight: 700;
            text-shadow: 0px 0px 15px rgba(0, 255, 255, 0.7);
        }}
        .glass-card {{
            background: rgba(10, 10, 25, 0.6); backdrop-filter: blur(10px) saturate(150%);
            -webkit-backdrop-filter: blur(10px) saturate(150%);
            border-radius: 20px; border: 1px solid rgba(255, 255, 255, 0.18);
            padding: 25px; margin-bottom: 25px;
        }}
        .stSelectbox label, .stNumberInput label {{
            font-family: 'Poppins', sans-serif; font-weight: 600; color: #e0e0e0 !important; font-size: 1.1rem;
        }}
        .stButton > button {{
            width: 100%; font-family: 'Poppins', sans-serif; font-size: 1.5rem; font-weight: 600;
            color: white; padding: 15px 0; background: linear-gradient(45deg, #00c6ff, #0072ff);
            border: none; border-radius: 12px; box-shadow: 0px 0px 20px rgba(0, 198, 255, 0.5);
            transition: all 0.3s ease-in-out;
        }}
        .score-display .score-value {{
            font-family: 'Poppins', sans-serif; font-size: 6rem; font-weight: 700;
            color: #ffffff; line-height: 1.1; text-shadow: 0px 0px 25px rgba(0, 255, 255, 0.7);
        }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


# Load the styles
load_css_and_background()

# --- App Layout ---

_, main_col, _ = st.columns([1, 3, 1])
with main_col:
    st.markdown('<div class="title-container" style="text-align: center;"><h1>🔮 Cricket Score Predictor 🏏</h1></div>',
                unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.header("🏟️ Match Setup")
        col1, col2 = st.columns(2)
        with col1:
            batting_team = st.selectbox('🏏 Batting Team', sorted(teams))
            logo_path = team_logos.get(batting_team)
            if logo_path:
                try:
                    st.image(logo_path, width=80)
                except Exception:
                    st.write("")
        with col2:
            bowling_team = st.selectbox('⚾ Bowling Team', sorted(teams))
            logo_path = team_logos.get(bowling_team)
            if logo_path:
                try:
                    st.image(logo_path, width=80)
                except Exception:
                    st.write("")

        # THIS IS THE CORRECTED LINE
        city = st.selectbox('📍 Venue City', sorted(cities))

        st.markdown('</div>', unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.header("📊 Current Match State")
        col3, col4 = st.columns(2)
        with col3:
            current_score = st.number_input('🔢 Current Score', min_value=0, value=75)
            overs = st.number_input('⏳ Overs Completed', min_value=5.0, max_value=19.5, value=10.0, step=0.1,
                                    format="%.1f")
        with col4:
            wickets = st.number_input('💥 Wickets Fallen', min_value=0, max_value=9, value=2)
            last_five = st.number_input("🔥 Runs in Last 5 Overs", min_value=0, value=40)
        st.markdown('</div>', unsafe_allow_html=True)

    if st.button('Predict Score 🚀'):
        if batting_team == bowling_team:
            st.error('Batting and Bowling teams cannot be the same.', icon="🚨")
        else:
            # Your original prediction logic
            overs_completed = int(overs)
            balls_in_current_over = int(round((overs - overs_completed) * 10))
            balls_left = 120 - (overs_completed * 6 + balls_in_current_over)

            pp = 0 if overs > 6 else 1
            wickets_left = 10 - wickets
            crr = current_score / overs if overs > 0 else 0

            Death_Overs = 1 if overs > 15 else 0
            Aggression_Mode = 1 if last_five >= 45 and overs > 10 else 0
            Pressure = 1 if crr <= 7 and overs > 5 else 0
            Top_Order = 1 if wickets <= 2 else 0
            Middle_Order = 1 if 3 <= wickets <= 5 else 0
            Lower_Order = 1 if 6 <= wickets <= 7 else 0
            Tail = 1 if wickets >= 8 else 0

            input_df = pd.DataFrame({
                'batting_team': [batting_team], 'bowling_team': [bowling_team], 'city': [city],
                'current_score': [current_score], 'pp': [pp], 'balls_left': [balls_left],
                'wickets_left': [wickets_left], 'crr': [crr], 'Top_Order': [Top_Order],
                'Middle_Order': [Middle_Order], 'Lower_Order': [Lower_Order], 'Tail': [Tail],
                'Pressure': [Pressure], 'Aggression_Mode': [Aggression_Mode],
                'last_five': [last_five], 'Death_Overs': [Death_Overs]
            })

            try:
                result = pipe.predict(input_df)
                predicted_score = int(result[0])
                score_html = f"""
                <div class="score-display" style="text-align: center;">
                    <p style="font-size: 1.5rem; color: #e0e0e0;">🎯 Predicted Final Score</p>
                    <p class="score-value">{predicted_score}</p>
                </div>
                """
                st.markdown(score_html, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")

