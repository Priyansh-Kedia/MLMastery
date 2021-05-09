# A value is normalise as follows
# y = (x - min) / (max - min)


from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler

series = read_csv('daily-min-temperatures.csv', header=0, index_col=0)
print(series.head())
print(series.shape)

values = series.values
print(values)
values = values.reshape((len(values), 1))

# train the normalization
scaler = MinMaxScaler(feature_range=(0,1))
scaler = scaler.fit(values)
