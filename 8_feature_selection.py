# Feature selection is a process where you automatically 
# select those features in your data that contribute most 
# to the prediction variable or output in which you are interested.

from pandas import read_csv
from numpy import set_printoptions
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier

# load data
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(filename, names = names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]

# feature extraction
test = SelectKBest(score_func=f_classif, k = 4)
fit = test.fit(X, Y)
# Select features according to the k highest scores.
# If k = 4, then this would select the 4 features which 
# contribute the most to the output. score_func is the
# function which is used to check what all features 
# contribute the most

# Selection of feature selection model depends on various
# factors, it can be seen 
# https://drive.google.com/file/d/1VuQDvpYipxJaadfgw3Fr-dYYvrbVwISq/view?usp=sharing

# summarise scores
set_printoptions(precision=3)
print(fit.scores_)
features = fit.transform(X)

# summarize selected features
print(features[0:5,:])


model = LogisticRegression(solver='lbfgs')
rfe = RFE(model, 3)
fit = rfe.fit(X, Y)
# Feature ranking with recursive feature elimination.
# The goal of recursive feature elimination (RFE) is to 
# select features by recursively considering smaller and 
# smaller sets of features. First, the estimator is trained 
# on the initial set of features and the importance of each 
# feature is obtained either through any specific attribute 
# or callable. Then, the least important features are pruned 
# from current set of features. That procedure is recursively 
# repeated on the pruned set until the desired number of 
# features to select is eventually reached.
print("Num Features: %d" % fit.n_features_)
print("Selected Features: %s" % fit.support_)
print("Feature Ranking: %s" % fit.ranking_)


# feature extraction
pca = PCA(n_components=3)
fit = pca.fit(X)
# summarize components
# You can learn how PCA works from here
# https://towardsdatascience.com/the-mathematics-behind-principal-component-analysis-fff2d7f4b643
print("Explained Variance: %s" % fit.explained_variance_ratio_)
print(fit.components_)

# feature extraction
model = ExtraTreesClassifier(n_estimators=10)
# This class implements a meta estimator that 
# fits a number of randomized decision trees 
# (a.k.a. extra-trees) on various sub-samples 
# of the dataset and uses averaging to improve 
# the predictive accuracy and control over-fitting.
model.fit(X, Y)
print(model.feature_importances_)