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

