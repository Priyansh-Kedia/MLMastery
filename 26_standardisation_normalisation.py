# A value is normalized as follows:
# y = (x – min) / (max – min)

#Normalisation
from numpy import asarray
from sklearn.preprocessing import MinMaxScaler

# define data
data = asarray([[100, 0.001],[8, 0.05],[50, 0.005],[88, 0.07],[4, 0.1]])
print(data)

scaler = MinMaxScaler()
scaled = scaler.fit_transform(data)
print(scaled)


#A value is standardized as follows:
# y = (x – mean) / standard_deviation
# Standardization
from numpy import asarray
from sklearn.preprocessing import StandardScaler
# define data
data = asarray([[100, 0.001],
				[8, 0.05],
				[50, 0.005],
				[88, 0.07],
				[4, 0.1]])
print(data)
# define standard scaler
scaler = StandardScaler()
# transform data
scaled = scaler.fit_transform(data)
print(scaled)


from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot as plt

dataset = read_csv('sonar.csv', header=None)

print(dataset.shape)
dataset.hist()
plt.show()


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from numpy import mean, std

data = dataset.values
# separate into input and output columns
X, y = data[:, :-1], data[:, -1]
# ensure inputs are floats and output is an integer label
X = X.astype('float32')

# Encode the output labels 
y = LabelEncoder().fit_transform(y.astype('str'))

# define and configure the model
model = KNeighborsClassifier()
# evaluate the model
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
# report model performance
print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))