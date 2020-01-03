import pandas as pd
import streamlit  as st
import joblib
import numpy as np


#create sidebar

#sidebar description
st.sidebar.header('Advertising Costs')
st.sidebar.subheader('How much are you spending?')
#cost inputs

tv = st.sidebar.number_input("TV Advertising Cost", min_value=0,max_value=300,value=150)
radio = st.sidebar.number_input("Radio Cost", min_value=0,max_value=50,value=10)
newspaper = st.sidebar.number_input("Newspaper Cost", min_value=0,max_value=250,value=100)

st.markdown('<style>h1{color: blue;}</style>', unsafe_allow_html=True)
st.markdown('<style>h3{color: DarkMagenta;}</style>', unsafe_allow_html=True)
st.title('Sales Forecasting')

st.write ('We demonstrate how we can forecast advertising sales.')


st.write ('______________________________________________________')
data = pd.read_csv("data/advertising_regression.csv")


#load model
saved_model=joblib.load('advertising_model.sav')

#predict
predicted_sales = round(saved_model.predict([[tv, radio, newspaper]])[0],2)


#print predictions"
st.subheader(f"Predicted sales is $ {predicted_sales}.")

st.write ('______________________________________________________')




if st.checkbox('Show Data?'):
    #data
    data

st.subheader('Radio Ad Cost Distribution')

#distribution of radio advertising
hist_values = np.histogram(data.radio, bins=300, range=(0,300))[0]

#show bar chart
st.bar_chart(hist_values)

st.subheader('TV Ad Cost Distribution')

#distribution of radio advertising
hist_values = np.histogram(data.TV, bins=300, range=(0,300))[0]

#show bar chart
st.bar_chart(hist_values)

st.subheader('Newspaper Ad Cost Distribution')

#distribution of radio advertising
hist_values = np.histogram(data.newspaper, bins=300, range=(0,300))[0]




#show bar chart
st.bar_chart(hist_values)



