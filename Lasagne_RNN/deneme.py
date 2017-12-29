__author__ = 'csburak'

import numpy as np
X = np.concatenate([np.random.uniform(size=(100, 55, 1)),
                        np.zeros((100, 55, 1))],
                       axis=-1)
print X
print len(X)
print X[0]
print X[0].shape
print X[1]
print X[1].shape

