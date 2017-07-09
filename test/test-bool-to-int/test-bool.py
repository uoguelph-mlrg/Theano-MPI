import theano
import theano.tensor as T
import numpy as np
np_a = np.array([90,40,3]).astype(np.int8)
np_b = np.array([90,77,1]).astype(np.int8)
a=theano.shared(np_a)
b=theano.shared(np_b)

c = T.neq(a,b).astype("int8")

print c.eval()
