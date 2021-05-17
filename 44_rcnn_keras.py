# Time series prediction problems are a difficult type of predictive 
# modeling problem.

# Unlike regression predictive modeling, time series also adds the 
# complexity of a sequence dependence among the input variables.

# A powerful type of neural network designed to handle sequence 
# dependence is called recurrent neural networks. The Long Short-Term 
# Memory network or LSTM network is a type of recurrent neural network 
# used in deep learning because very large architectures can be 
# successfully trained.

import pandas as pd
import matplotlib.pyplot as plt

# engine: Parser engine to use. The C engine is faster while the 
# python engine is currently more feature-complete.
dataset = pd.read_csv("international-airlines-passengers.csv", usecols=[1],engine="python")

plt.plot(dataset)
plt.show()

# The Long Short-Term Memory network, or LSTM network, is a recurrent 
# neural network that is trained using Backpropagation Through Time and 
# overcomes the vanishing gradient problem.


