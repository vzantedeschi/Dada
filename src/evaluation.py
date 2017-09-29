import numpy as np

from sklearn.metrics import accuracy_score

def alpha_variance(nodes):
    return np.var([n.alpha for n in nodes], axis=0)

def mean_accuracy(nodes):
    return np.mean([accuracy_score(np.inner(n.sample, n.clf), n.labels) for n in nodes])