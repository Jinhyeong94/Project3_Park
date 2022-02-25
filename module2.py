# %%
# import
from traceback import print_exception
import pandas as pd, numpy as np
from sqlalchemy import column
import yfinance as yf  # Yahoo Finance
from pandas_datareader import data as pdr 
import ta  # Technical Analysis
import matplotlib.pyplot as plt
import hvplot.pandas

# %%
# Check tickers
all_tickers = input("enter first stock:")
print(all_tickers)
# %%
# Show dataframe
yf.pdr_override()

data = pdr.get_data_yahoo(all_tickers,
    start="2020-01-01",
    end="2022-01-31"
)
data['Close']

# %%
# dividend

KO = yf.Ticker(all_tickers)
KO.dividends

data_dividends = KO.dividends
print(type(data_dividends))

# %%

df_dividends = pd.DataFrame(data_dividends)
print(df_dividends)

# %%
# recommandation

df_recommendations = KO.recommendations
df_recommendations = pd.DataFrame(df_recommendations)
print(df_recommendations)
# %%
# Indicators
bol_h = ta.volatility.bollinger_hband(data['Close'])
bol_l = ta.volatility.bollinger_lband(data['Close'])
rsi = ta.momentum.rsi(
    data['Close'],
    window = 14
    )
plt.plot(data['Close'],c='k')
plt.plot(bol_h,c='r')
plt.plot(bol_l,c='g')
plt.plot(rsi,c='b')
# %%
print(rsi)
# %%
# Set Index
data = pd.DataFrame(data)
print(data)
# %%
signals_df = data.loc[:, ["Close"]].copy()

# Set the short window and long windows
SMA_window = 50
# Short term // EMA26, EMA12 for MACD
# MACD = EMA12 - EMA26
EMA26_window = 26
EMA12_window = 12


# Create a short window SMA
signals_df["SMA50"] = signals_df["Close"].rolling(window=SMA_window).mean()

# Create a short window EMA
signals_df["EMA12"] = signals_df["Close"].ewm(span=EMA12_window).mean()

# Create a short window EMA
signals_df["EMA26"] = signals_df["Close"].ewm(span=EMA26_window).mean()


# Review the DataFrame
signals_df.iloc[45:55, :]

# %%
df_ema12 = signals_df['EMA12']
df_ema26 = signals_df['EMA26']

df_MACD = df_ema12 - df_ema26
df_MACD = pd.DataFrame(df_MACD)
df_MACD.columns = ['MACD']
df_MACD
# %% connect Close + SMA50 + MACD
df3 = signals_df.loc[:, ["Close", "SMA50"]]

result = pd.concat([df3,df_MACD], axis = 1)
result
# %%
# LSTM

from numpy.random import seed

seed(1)
from tensorflow import random

random.set_seed(2)
# %%
def window_data(result, window, feature_col_number, target_col_number):
    """
    This function accepts the column number for the features (X) and the target (y).
    It chunks the data up with a rolling window of Xt - window to predict Xt.
    It returns two numpy arrays of X and y.
    """
    X = []
    y = []
    for i in range(len(result) - window):
        features = result.iloc[i : (i + window), feature_col_number]  # [1:6, 2] = X
        target = result.iloc[(i + window), target_col_number]  # [6, 2] = y
        X.append(features)
        y.append(target)
    return np.array(X), np.array(y).reshape(-1, 1)
# %%
# Creating the features (X) and target (y) data using the window_data() function.
window_size = 5

feature_column = 2
target_column = 2
X, y = window_data(result, window_size, feature_column, target_column)
print (f"X sample values:\n{X[:5]} \n")
print (f"y sample values:\n{y[:5]}")
# %%
# Use 70% of the data for training and the remainder for testing
split = int(0.7 * len(X))
X_train = X[: split]
X_test = X[split:]
y_train = y[: split]
y_test = y[split:]
split
# %%
# Use the MinMaxScaler to scale data between 0 and 1.
from sklearn.preprocessing import MinMaxScaler

# Create a MinMaxScaler object
scaler = MinMaxScaler()

# Fit the MinMaxScaler object with the training feature data X_train
scaler.fit(X_train)

# Scale the features training and testing sets
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Fit the MinMaxScaler object with the training target data y_train
scaler.fit(y_train)

# Scale the target training and testing sets
y_train = scaler.transform(y_train)
y_test = scaler.transform(y_test)
# %%
print(type(X_train))
print(X_train.shape)
print(X_train.shape[0])  # = rows
print(X_train.shape[1])  # = columns
# %%
# Reshape the features for the model
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))  # last param is number of elements, e.g. [[elem_1], [elem_2]]
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
print (f"X_train sample values:\n{X_train[:5]} \n")
print (f"X_test sample values:\n{X_test[:5]}")
# %%
# Import required Keras modules
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout  
# %%
# Define the LSTM RNN model.
model = Sequential()

number_units = 5  # equals the time window
dropout_fraction = 0.2

# Layer 1
model.add(LSTM(
    units=number_units,
    # except for final layer, each time we add a new LSTM layer, we must set return_sequences=True
    # it just lets Keras know to connect each layer
    return_sequences=True,
    input_shape=(X_train.shape[1], 1))
    )
model.add(Dropout(dropout_fraction))
# Layer 2
model.add(LSTM(units=number_units, return_sequences=True))
model.add(Dropout(dropout_fraction))
# Layer 3
model.add(LSTM(units=number_units))
model.add(Dropout(dropout_fraction))
# Output layer
model.add(Dense(1))  
# %%
# Compile the model
model.compile(optimizer="adam", loss="mean_squared_error")
# %%
# Summarize the model
model.summary()
# %%
# Train the model
model.fit(X_train, y_train, epochs=10, shuffle=False, batch_size=1, verbose=1)
# %%
# Evaluate the model
model.evaluate(X_test, y_test)
# %%
# Make some predictions
predicted = model.predict(X_test)