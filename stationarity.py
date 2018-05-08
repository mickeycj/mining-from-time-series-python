import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.stattools import adfuller

# Test the stationarity of the time series.
def test_stationarity(time_series, window_title):

    # Create the statistics.
    dataframe = pd.DataFrame(time_series)
    rolling_mean = dataframe.rolling(window=12, center=False).mean()
    rolling_std = dataframe.rolling(window=12, center=False).std()

    # Show the result of Dickey-Fuller Test.
    print('Results of Dicky-Fuller Test:')
    df_test = adfuller(time_series, autolag='AIC')
    df_output = pd.Series(df_test[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in df_test[4].items():
        df_output['Critical Value (%s)' % key] = value
    print(df_output)

    # Plot the statistics.
    fig = plt.figure(0)
    fig.canvas.set_window_title(window_title)
    plt.title('Rolling Mean & Standard Deviation')
    plt.legend(loc='best')
    plt.plot(time_series, color='blue', label='Original')
    plt.plot(rolling_mean, color='red', label='Rolling Mean')
    plt.plot(rolling_std, color='black', label='Rollding Std')
    plt.show()
