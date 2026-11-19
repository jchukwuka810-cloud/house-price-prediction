import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="House Price Predictor", page_icon="🏠")


@st.cache_resource
def load_files():
    model = joblib.load("house_price_model.joblib")
    scaler = joblib.load("scaler.joblib")
    features = joblib.load("features.joblib")
    return model, scaler, features


model, scaler, features = load_files()

st.title("🏠 California House Price Predictor")
st.write(
    "Enter the details of a California neighbourhood and click the button "
    "to get an estimated median house price."
)

col1, col2 = st.columns(2)

with col1:
    income = st.number_input(
        "Median household income ($ per year)",
        min_value=5000, max_value=150000, value=50000, step=1000,
    )
    house_age = st.slider("Median house age (years)", 1, 52, 25)
    ave_rooms = st.slider("Average rooms per house", 2.0, 12.0, 5.5, 0.1)
    ave_bedrms = st.slider("Average bedrooms per house", 0.5, 3.0, 1.1, 0.1)

with col2:
    ave_occup = st.slider("Average people per house", 1.0, 6.0, 3.0, 0.1)
    latitude = st.slider("Latitude", 32.5, 42.0, 34.0, 0.01)
    longitude = st.slider("Longitude", -124.4, -114.3, -118.0, 0.01)

if st.button("Predict price"):
    row = pd.DataFrame(
        [[income / 10000, house_age, ave_rooms, ave_bedrms,
          ave_occup, latitude, longitude]],
        columns=features,
    )
    row_scaled = scaler.transform(row)
    prediction = model.predict(row_scaled)[0] * 100000

    st.success(f"Estimated median house price: ${prediction:,.0f}")
    st.caption(
        "Typical error is about $31,000. Reliable only for neighbourhoods "
        "below $500,000."
    )