# House Price Prediction

**Goal:** This project predicts the median house price of a California
neighbourhood using features such as median income, house age and number
of rooms. It uses Linear Regression, trained on the California Housing
dataset (20,640 samples, 8 features). The model is deployed as a Streamlit
web app where a user enters the features and sees the predicted price.

## Results

I trained two models and compared them on 3,909 neighbourhoods the model
had never seen.

| Model | MAE | R2 |
|---|---|---|
| Linear Regression | 0.446 | 0.622 |
| Random Forest (final) | 0.309 | 0.777 |

**In plain words:** The final Random Forest's price estimate is, on average,
about $31,000 away from the real value (MAE 0.309 x $100,000). It explains
about 78% of the differences in house prices (R2 0.777).

It works best for mid-priced neighbourhoods ($50,000 to $250,000). For
expensive areas (above about $300,000), it tends to guess too low.

**Limitation:** Prices of $500,000 or more were capped in the original data,
so those rows were removed. The model is only reliable for neighbourhoods
below $500,000.

## Live App

https://josh-house-price-prediction.streamlit.app/

## How to run locally

1. Clone the repository and open the folder.
2. Create and activate a virtual environment.
3. Install the requirements: `pip install -r requirements.txt`
4. Start the app: `streamlit run streamlit_app.py`

## Files

- `house_price.ipynb`: data exploration, cleaning, training and evaluation
- `streamlit_app.py`: the web app
- `house_price_model.joblib`, `scaler.joblib`, `features.joblib`: the saved model, scaler and feature list
- `requirements.txt`: libraries needed to run the app