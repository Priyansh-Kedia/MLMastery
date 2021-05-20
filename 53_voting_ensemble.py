from numpy import mean, std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

X,Y = make_classification(random_state=1)

# configure the models to use in the ensemble
models = [('lr', LogisticRegression()), ('nb', GaussianNB())]

# voting: If ‘hard’, uses predicted class labels for majority rule voting. 
# Else if ‘soft’, predicts the class label based on the argmax of the sums 
# of the predicted probabilities, which is recommended for an ensemble of 
# well-calibrated classifiers.
model = VotingClassifier(models, voting='soft')

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

n_scores = cross_val_score(model, X, Y, scoring='accuracy', cv=cv, n_jobs=1)

# report ensemble performance
print('Mean Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))