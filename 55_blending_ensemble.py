# Blending ensemble for classification 
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from numpy import hstack
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

def get_dataset():
    X,Y = make_classification(n_samples=10000, n_features=20, n_informative=15, n_redundant=5, random_state=7)
    print(X.shape, Y.shape)
    return X,Y

# list of base models
def get_models():
    models = list()
    models.append(("lr", LogisticRegression()))
    models.append(("knn",KNeighborsClassifier()))
    models.append(('cart', DecisionTreeClassifier()))
    models.append(('svm', SVC()))
    models.append(('bayes', GaussianNB()))
    return models

# fit the blending ensemble
def fit_ensemble(models, X_train, X_val, Y_train, Y_val):
    meta_x = list()
    for name, model in models:
        model.fit(X_train, Y_train)
        yhat = model.predict(X_val)
        yhat = yhat.reshape(len(yhat), 1)
        meta_x.append(yhat)

    # Stack arrays in sequence horizontally (column wise).
    meta_x = hstack(meta_x)
    blender = LogisticRegression()
    blender.fit(meta_x, Y_val)
    return blender

def predict_ensemble(models, blender, X_test):
    meta_X = list()
    for name, model in models:
        yhat = model.predict(X_test)
	    # reshape predictions into a matrix with one column
        yhat = yhat.reshape(len(yhat), 1)
	    # store prediction
        meta_X.append(yhat)

    meta_X = hstack(meta_X)
    return blender.predict(meta_X)

X,Y = get_dataset()

# split dataset into train and test sets
X_train_full, X_test, Y_train_full, Y_test = train_test_split(X, Y, test_size=0.5, random_state=1)

# split training set into train and validation sets
X_train, X_val, Y_train, Y_val = train_test_split(X_train_full, Y_train_full, test_size=0.33, random_state=1)

# summarize data split
print('Train: %s, Val: %s, Test: %s' % (X_train.shape, X_val.shape, X_test.shape))

models = get_models()

blender = fit_ensemble(models, X_train, X_val, Y_train, Y_val)

yhat = predict_ensemble(models, blender, X_test)

# evaluate predictions
score = accuracy_score(Y_test, yhat)
print('Blending Accuracy: %.3f' % (score*100))

