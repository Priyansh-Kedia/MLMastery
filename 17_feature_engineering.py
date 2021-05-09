# The goal of feature engineering is to provide strong and 
# ideally simple relationships between new input features 
# and the output feature for the supervised learning algorithm to model.

from pandas import read_csv
from pandas import DataFrame
import numpy as np
from matplotlib import pyplot as plt

series = read_csv('daily-min-temperatures.csv', index_col=0, header=0, parse_dates = True, squeeze=True)
dataframe = DataFrame()
dataframe['month'] = [series.index[i].month for i in range(len(series))]
dataframe['day'] = [series.index[i].day for i in range(len(series))]
dataframe['temperature'] = [series[i] for i in range(len(series))]
print(dataframe)

print(series[0]) # This will print the temperature at first position
print(series.index[0].month) # This will print the month at first position
print(series.index[0].day) # This will print the day at first position
print(series.index[0]) # This will print the entire data at first position including the index

series.plot()
plt.show()


# Lag features are the classical way that time series forecasting problems 
# are transformed into supervised learning problems.
# The simplest approach is to predict the value at the next time (t+1) 
# given the value at the previous time (t-1).

# The Pandas library provides the shift() function to help create these 
# shifted or lag features from a time series dataset. Shifting the dataset 
# by 1 creates the t-1 column, adding a NaN (unknown) value for the first 
# row. The time series dataset without a shift represents the t+1.

from pandas import concat

series = read_csv('daily-min-temperatures.csv', header=0, index_col=0)
temps = DataFrame(series.values)
dataframe = concat([temps.shift(1), temps], axis=1)
# shift(1) tells to shift by 1, axis defines the axis to be shifted
dataframe.columns = ['t-1','t+1']
print(dataframe.head(5))
# This is also known as sliding window method, where the width of the 
# window is 1

# If last 3 values need to be included, then something like this would be done
dataframe = concat([temps.shift(3), temps.shift(2), temps.shift(1), temps], axis=1)
dataframe.columns = ['t-3', 't-2', 't-1', 't+1']
print(dataframe.head(5))



# Rolling window statistics
temps = DataFrame(series.values)
shifted = temps.shift(1)
window = shifted.rolling(window=2)
# rolling window specifies the window to be taken when using rolling window.
# So for t = 2, values will be taken for t = 1, and t = 0.
means = window.mean()
dataframe = concat([means, temps], axis=1)
dataframe.columns = ['mean(t-2,t-1)', 't+1']
print(dataframe.head(5))





