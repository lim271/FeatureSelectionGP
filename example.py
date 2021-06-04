from time import time
import numpy as np
from fsgp import FeatureSelectionGPR

# np.random.seed(7)

def compute_accuracy():
    X = 1. / (np.arange(1, 51) + np.arange(0, 200)[:, np.newaxis])
    h = np.random.randn(50,1)
    y = X.dot(h).reshape((-1, 1)) 
    X = np.concatenate((X, np.random.randn(200, 50)), axis=1)

    fsgp = FeatureSelectionGPR(n_restarts_optimizer=0, regularization_param=1.0)
    tic = time()
    fsgp.fit(X, y)
    toc = time()
    # print('h=', h.reshape((-1)))
    # print('y=', y.reshape((-1)))
    # print('w=', fsgp.kernel_.k1.k2.length_scale)
    tp = 0
    tn = 0
    for idx, param in enumerate(fsgp.kernel_.k1.k2.length_scale):
        if idx < 50:
            if param!=0.0:
                tp += 1
        else:
            if param==0.0:
                tn += 1

    return toc-tic, tp, tn

if __name__=="__main__":
    n = 10
    avg_t = 0
    acc = 0
    for _ in range(n):
        t, tp, tn = compute_accuracy()
        avg_t += t
        acc   += ((tp + tn) / 100.0)
    acc   /= float(n)
    avg_t /= float(n)
    print("Accuracy:", acc, "Time:", avg_t)