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
        return np.inner(x, self.h)

class RandomClassifier(WeakClassifier):

    def predict(self, x):
        r = np.random.randint(2, size=x.shape[0])
        r[r==0] = -1
        return r
