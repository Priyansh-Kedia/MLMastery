# Python version
import sys
print('Python: {}'.format(sys.version))
# scipy
import scipy
print('scipy: {}'.format(scipy.__version__))
# numpy
import numpy
print('numpy: {}'.format(numpy.__version__))
# matplotlib
import matplotlib
print('matplotlib: {}'.format(matplotlib.__version__))
# pandas
import pandas
print('pandas: {}'.format(pandas.__version__))
# scikit-learn
import sklearn
print('sklearn: {}'.format(sklearn.__version__))





# import statements for models
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


# load the dataset
url = "iris.csv"
col_names = ['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm','Species']
dataset = read_csv(url, usecols = col_names)
# print(dataset)

# printing first 20 data points
print(dataset.head(20))

# describe the dataset
print(dataset.describe())

# class distribution
print(dataset.groupby('Species').size())


# data visualisation
# dataset.plot(kind = 'box', subplots = True, layout = (2,2), sharex = False, sharey = False)
# plt.show()

# dataset.hist()
# plt.show()

# scatter_matrix(dataset)
# plt.show()

array = dataset.values
X = array[:,0:4]
y = array[:,4]
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size = 0.2, random_state = 1)
# In this case, random state basically specifies that your data will be split in a specific order always
# For example, the order you will get in random_state=0 remain same. After that if you execute
# random_state=5 and again come back to random_state=0 you'll get the same order.

models = []
models.append(('LR', LogisticRegression(solver='liblinear',multi_class='ovr')))
# solver can take {‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’}, 
# and default=’lbfgs’. This specifies the algorithm to use for optimisation.
# For small datasets, ‘liblinear’ is a good choice, whereas ‘sag’ and ‘saga’ are 
# faster for large ones. For multiclass problems, only ‘newton-cg’, ‘sag’, ‘saga’
#  and ‘lbfgs’ handle multinomial loss; ‘liblinear’ is limited to one-versus-rest schemes.

# multi_class is {‘auto’, ‘ovr’, ‘multinomial’}, default=’auto’
# If the option chosen is ‘ovr’, then a binary problem is fit for 
# each label. For ‘multinomial’ the loss minimised is the multinomial 
# loss fit across the entire probability distribution, even when the 
# data is binary. ‘multinomial’ is unavailable when solver=’liblinear’.
# ‘auto’ selects ‘ovr’ if the data is binary, or if solver=’liblinear’, 
# and otherwise selects ‘multinomial’.

models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))

models.append(('SVM', SVC(gamma='auto')))
# {‘scale’, ‘auto’} or float, default=’scale’
# gamma is a parameter for non linear hyperplanes. 
# The higher the gamma value it tries to exactly fit the training data set.
# One more parameter that can be passed is, kernels = [‘linear’, ‘rbf’, ‘poly’]
# Using ‘linear’ will use a linear hyperplane (a line in the case of 2D data). 
# ‘rbf’ and ‘poly’ uses a non linear hyper-plane.


results = []
names = []
for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    # Provides train/test indices to split data in train/test sets.

    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    # Evaluate a score by cross-validation

    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

plt.boxplot(results, labels=names)
plt.title("Algorithm comparison")
plt.show()

model = SVC(gamma='auto')
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)

# Evaluate predictions
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))