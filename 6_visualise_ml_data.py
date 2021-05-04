import matplotlib.pyplot as plt
import pandas as pd
import numpy
from pandas.plotting import scatter_matrix

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = pd.read_csv(url, names=names)

# Histogram
data.hist()

# Density Plot
data.plot(kind='density', subplots=True, layout=(3,3), sharex=False)

# Box and Whisker
data.plot(kind='box', subplots=True, layout=(3,3), sharex=False, sharey=False)


# Correlations matrix plot
correlations = data.corr()
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = numpy.arange(0,9,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names)
ax.set_yticklabels(names)


# Scatter Plot
scatter_matrix(data)

plt.show()