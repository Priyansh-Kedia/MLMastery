from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
from sklearn.ensemble import BaggingClassifier

X, Y = make_classification(random_state=1)