import pandas as pd

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = pd.read_csv(url, names = names)

print(data.head(20))
print(data.shape)
print(data.dtypes)

description = data.describe()
print(description)

class_counts = data.groupby('class').size()
print(class_counts)

pd.set_option('display.width', 100)
pd.set_option('precision', 3)
correlations = data.corr(method='pearson')
print(correlations)

skew = data.skew()
print(skew)
# Skewness is a measure of the asymmetry of the probability 
# distribution of a real-valued random variable about its mean