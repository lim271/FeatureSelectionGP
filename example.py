import numpy as np
import matplotlib.pyplot as plt

from fsgp import FeatureSelectionGPR


X = 1. / (np.arange(1, 11) + np.arange(0, 10)[:, np.newaxis])
h = np.random.randn(10,1)
y = X.dot(h).reshape((-1, 1)) 

fsgp = FeatureSelectionGPR(n_restarts_optimizer=100, regularization_param=1.0)
fsgp.fit(X, y)
print('h=', h.reshape((-1)))
print('y=', y.reshape((-1)))
#print('w=', fsgp.kernel_.k1.k2.length_scale)
print('w=', fsgp.kernel_.k1.length_scale)
