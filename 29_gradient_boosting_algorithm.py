# The idea of boosting came out of the idea of whether a 
# weak learner can be modified to become better.

# A weak hypothesis or weak learner is defined as one whose 
# performance is at least slightly better than random chance.

# Hypothesis boosting was the idea of filtering observations, 
# leaving those observations that the weak learner can handle 
# and focusing on developing new weak learners to handle the 
# remaining difficult observations.


# AdaBoost
# The weak learners in AdaBoost are decision trees with a single 
# split, called decision stumps for their shortness.
# AdaBoost works by weighting the observations, putting more weight 
# on difficult to classify instances and less on those already handled 
# well. New weak learners are added sequentially that focus their 
# training on the more difficult patterns.

# Gradient boosting involves three elements:
# A loss function to be optimized.
# A weak learner to make predictions.
# An additive model to add weak learners to minimize the loss function.

# Decision trees are used as the weak learner in gradient boosting.







