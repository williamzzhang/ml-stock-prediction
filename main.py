from matplotlib.pyplot import text
import streamlit as st 
from datetime import date

import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

# Creating dashboard elements and initializing variables for data loading
START = "2017-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title("Stock Prediction Prophet")

stocks = ("AAPL", "GOOGL", "MSFT", "GME", "TSLA")
selected_stock = st.selectbox("Select dataset for prediction", stocks)

n_years = st.slider("Years of prediction:", 1, 5)
period = n_years * 365

# Data loading and caching
@st.cache(allow_output_mutation=True)
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text("Loading data...")
data = load_data(selected_stock)
data_load_state.text("Data loaded!")

# Stripping timezones as fbprophet requires no timezones before fitting
data['Date'] = data['Date'].dt.tz_localize(None)

st.subheader('Raw data')
st.write(data.tail())

# Plotting adjustable time series
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='Opening Price'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='Closing Price'))
    fig.layout.update(title_text="Time Series Data (Adjust bottom slider to change time period)", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()

# Forecasting using fbprophet
df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

p = Prophet()
p.fit(df_train)
future = p.make_future_dataframe(periods=period)
forecast = p.predict(future)

st.subheader('Forecast data')
st.write(forecast.tail())

st.write('Forecast data')
fig1 = plot_plotly(p, forecast)
st.plotly_chart(fig1)

st.write('Forecast components')
fig2 = p.plot_components(forecast)
st.write(fig2)