import streamlit as st
import pickle
import pandas as pd
import numpy as np
import base64
import xgboost
#st.set_page_config()
from PIL import Image

img = Image.open('ICCMT20WC2021_2020-symbol.png')
st.set_page_config(page_title='Cricket Score Predictor', page_icon=img)
img2 = Image.open('BCCI.png')
pipe = pickle.load(open('pipe (4).pkl', 'rb'))
print(xgboost.__version__)
teams = [
    'Australia',
    'India',
    'Bangladesh',
    'New Zealand',
    'South Africa',
    'England',
    'West Indies',
    'Afghanistan',
    'Pakistan',
    'Sri Lanka'
]

cities = ['Colombo',
     'Mirpur',
     'Johannesburg',
     'Dubai',
     'Auckland',
     'Cape Town',
     'London',
     'Pallekele',
     'Barbados',
     'Sydney',
     'Melbourne',
     'Durban',
     'St Lucia',
     'Wellington',
     'Lauderhill',
     'Hamilton',
     'Centurion',
     'Manchester',
     'Abu Dhabi',
     'Mumbai',
     'Nottingham',
     'Southampton',
     'Mount Maunganui',
     'Chittagong',
     'Kolkata',
     'Lahore',
     'Delhi',
     'Nagpur',
     'Chandigarh',
     'Adelaide',
     'Bangalore',
     'St Kitts',
     'Cardiff',
     'Christchurch',
     'Trinidad']


#st.image(img2, width=55)
col1, col2 = st.columns(2)
with col1:
    st.image(img, width=215)
with col2:
    st.title('Cricket Score Predictor')

img2 = Image.open('BCCI.png')
img3 = Image.open('eng.png')

#with st.sidebar.container():

col3, col4 = st.columns(2)

with col3:
    batting_team = st.selectbox('Select batting team', sorted(teams))
    if batting_team=="India":
        st.image(img2, width=105)
    elif batting_team=="Australia":
        st.image("AUS.png", width=105)
    elif batting_team == "Pakistan":
        st.image("PCB.png", width=105)
    elif batting_team == "West Indies":
        st.image("WI.png", width=105)
    elif batting_team == "South Africa":
        st.image("RSA.png", width=105)
    elif batting_team == "Sri Lanka":
        st.image("SL.png", width=105)
    elif batting_team == "Afghanistan":
        st.image("AFG.png", width=105)
    elif batting_team == "Bangladesh":
        st.image("BAN1.png", width=105)
    elif batting_team == "England":
        st.image("England.png", width=105)
    elif batting_team == "New Zealand":
        st.image("NZ1.png", width=105)
    #, use_column_width = 'never'

with col4:
    bowling_team = st.selectbox('Select bowling team', sorted(teams))
    if bowling_team =="India":
        st.image(img2, width=105)
    elif bowling_team =="Australia":
        st.image("AUS.png", width=105)
    elif bowling_team =="Pakistan":
        st.image("PCB.png", width=105)
    elif bowling_team =="West Indies":
        st.image("WI.png", width=105)
    elif bowling_team =="Sri Lanka":
        st.image("SL.png", width=105)
    elif bowling_team == "Afghanistan":
        st.image("AFG.png", width=105)
    elif bowling_team == "Bangladesh":
        st.image("BAN1.png", width=105)
    elif bowling_team == "England":
        st.image("England.png", width=105)
    elif bowling_team == "New Zealand":
        st.image("NZ1.png", width=105)
    elif bowling_team == "South Africa":
        st.image("RSA.png", width=105)
col5,col6 = st.columns(2)
with col5:
    city = st.selectbox('Select city', sorted(cities))
with col6:
    current_score = st.number_input('Current Score', min_value=0, max_value=714, value=0, step=1)

col7, col8, col9 = st.columns(3)

with col7:
    overs = st.number_input('Overs', min_value=5, max_value=19, value=5, step=1)

with col8:
    wickets = st.number_input('Wickets Fallen', min_value=0, max_value=9, value=0, step=1)

with col9:
    last_five = st.number_input("Runs scored in last 5 overs", min_value=0, max_value=180, value=0, step=1)


if st.button('Predict Score'):
    balls_left = 120 - (overs * 6)
    if balls_left<=84:
        pp=0
    else:
        pp=1
    wickets_left = 10 - wickets
    crr = current_score/overs
    if wickets_left<5:
        Main_Batsmen = 0
    else:
        Main_Batsmen = 1
    if overs == 10:
        if current_score <= 90:
            Upperhand = 0
        else:
            Upperhand = 1
    if overs<=15:
        Death_Overs = 0
    else:
        Death_Overs = 1
    if last_five>=45:
        Aggression_Mode=1
    else:
        Aggression_Mode = 0
    if crr<=7:
        Pressure=1
    else:
        Pressure=0
    if wickets_left>=9:
        Top_Order=1
    else:
        Top_Order=0
    if wickets_left >=7 and wickets_left<=8:
        Middle_Order=1
    else:
        Middle_Order=0
    if wickets_left<=6 and wickets_left >=5:
        Lower_Order=1
    else:
        Lower_Order=0
    if wickets_left<=4:
        Tail=1
    else:
        Tail=0

    if batting_team == bowling_team:
        #st.header("Batting and Bowling Team cannot be the same")
        st.error('Batting and Bowling Team cannot be the same', icon="🚨")
    else: #'batting_team','bowling_team','city','current_score', 'pp', 'balls_left','wickets_left','crr', 'Main_Batsmen', 'Upperhand', 'last_five', 'Death_Overs','runs_x']]
        input_df = pd.DataFrame(
        {'batting_team': [batting_team], 'bowling_team': [bowling_team], 'city': city, 'current_score': [current_score],
         'pp':[pp], 'balls_left': [balls_left], 'wickets_left': [wickets_left], 'crr': [crr], 'Top_Order':[Top_Order], 'Middle_Order':[Middle_Order], 'Lower_Order':[Lower_Order], 'Tail':[Tail], 'Pressure':[Pressure], 'Aggression_Mode':[Aggression_Mode], 'last_five': [last_five],
         'Death_Overs': [Death_Overs]})
        result = pipe.predict(input_df)
        st.header("Predicted Score : " + str(int(result[0])))