from abc import ABCMeta, abstractmethod

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score

# ------------------------------------------------------------------ CLASSIFIERS
class WeakClassifier(BaseEstimator):

    def __init__(self):
        pass

    @abstractmethod
    def predict(self, x):
        pass

    def score(self, x, y, sample_weight=None):

        y_pred = self.predict(x)
        return accuracy_score(y, y_pred, sample_weight=sample_weight)

class LinearClassifier(WeakClassifier):

    def __init__(self, d, h=None):

        if h is None:
            self.h = np.zeros((1, d))
        else:
            self.h = h

    def predict(self, x):
        return np.sign(np.inner(x, self.h))[:, None]

class RandomClassifier(WeakClassifier):

    def predict(self, x):
        r = np.random.randint(2, size=x.shape[0])
        r[r==0] = -1
        return r

class StumpClassifier(WeakClassifier):

    def __init__(self, index, threshold):

        self.id = index
        self.thr = threshold

    def predict(self, x):
        return 1 - 2*(x[:, self.id] > self.thr)


# ---------------------------------------------------------------- BASE CLFS GENERATION

def get_basis(n, d):

    vectors = np.eye(n, d)
    base_clfs = [LinearClassifier(d, v) for v in vectors]

    return base_clfs

def get_double_basis(n, d):

    vectors = np.append(np.eye(n // 2, d), -np.eye(n // 2, d), axis=0) 
    base_clfs = [LinearClassifier(d, v) for v in vectors]

    return base_clfs