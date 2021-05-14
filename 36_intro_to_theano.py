# Theano is a Python library for fast numerical computation that can be run 
# on the CPU or GPU.

import theano
from theano import tensor

a = tensor.dscalar()
b = tensor.dscalar()

c = a + b

f = theano.function([a,b], c)

assert 4.0 == f(1.5,2.5)
