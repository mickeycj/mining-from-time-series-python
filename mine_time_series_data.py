import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.pylab import rcParams
from stationarity import test_stationarity
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller

rcParams['figure.figsize'] = 15, 6

def set_window_title(window_title):
    fig = plt.figure(0)
    fig.canvas.set_window_title(window_title)

# Load the dataset from CSV file.
date_parser = lambda dates: pd.datetime.strptime(dates, '%Y-%m')
data = pd.read_csv('AirPassengers.csv', parse_dates=['Month'], index_col='Month', date_parser=date_parser)
time_series = data['#Passengers']

# Plot the time series
set_window_title('Air Passengers Time Series')
plt.plot(time_series, color='blue')
plt.show()

# Plot the raw data's Dickey-Fuller test.
test_stationarity(time_series, 'Original Time Series Statistics')

# Take the difference of the time series.
time_series_log = np.log(time_series)
time_series_log_diff = time_series_log - time_series_log.shift()
time_series_log_diff.dropna(inplace=True)

# Plot the pre-processed data's Dickey-Fuller test.
test_stationarity(time_series_log_diff, 'Pre-processed Time Series Statistics')

# Create ARIMA model.
model = ARIMA(time_series_log, order=(2, 1, 2))
results = model.fit(disp=-1)

# Plot the predicted time series.
set_window_title('Predicted Time Series')
plt.title('RSS: %.4f' % sum((results.fittedvalues - time_series_log_diff)**2))
plt.plot(time_series_log_diff, color='blue')
plt.plot(results.fittedvalues, color='red')
plt.show()

# Convert the predicted times series back to the original scale.
predictions_diff = pd.Series(results.fittedvalues, copy=True)
predictions_diff_cumsum = predictions_diff.cumsum()
predictions_log = pd.Series(time_series_log.ix[0], index=time_series_log.index)
predictions_log = predictions_log.add(predictions_diff_cumsum, fill_value=0)
predictions = np.exp(predictions_log)

# Plot the predicited time series in the original scale.
set_window_title('Predicted Time Series - Original Scale')
plt.title('RMSE: %.4f' % np.sqrt(sum((predictions - time_series)**2) / len(time_series)))
plt.plot(time_series, color='blue')
plt.plot(predictions, color='red')
plt.show()