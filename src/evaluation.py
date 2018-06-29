import warnings

from math import log
import numpy as np

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

from classification import RandomClassifier

def edges(nodes, *args):
    try:
        return [len(n.neighbors) for n in nodes]
    except:
        return None

def alpha_variance(nodes, *args):
    return np.around(np.mean(np.var([n.alpha for n in nodes], axis=0)), decimals=10)

def loss(nodes, *args):
    return np.sum([log(np.mean(n.compute_weights(distr=False))) for n in nodes])

def central_loss(nodes, *args):
    return log(np.mean(np.concatenate([n.compute_weights(distr=False) for n in nodes])))

def train_accuracies(nodes, *args):
    """ returns training accuracies per node"""
    train_acc = []

    for n in nodes:
        train_acc.append(accuracy_score(n.predict(n.sample), n.labels))

    return train_acc

def test_accuracies(nodes, *args):
    """ returns testing accuracies per node"""

    test_acc = []
    try:

        for n in nodes:
            test_acc.append(accuracy_score(n.predict(n.test_sample), n.test_labels))

    except:
        warnings.warn("Test sets not set in nodes.")
        
    return test_acc

def central_train_accuracy(nodes, *args):
    """ returns training accuracy """

    predictions, labels = [], []
    for n in nodes:
        predictions.append(n.predict(n.sample))
        labels.append(n.labels)

    train_acc = accuracy_score(np.concatenate(predictions), np.concatenate(labels))

    return train_acc

def central_test_accuracy(nodes, *args):
    """ returns testing accuracy """

    try:
        predictions, labels = [], []
        for n in nodes:
            predictions.append(n.predict(n.test_sample))
            labels.append(n.test_labels)
        
        test_acc = accuracy_score(np.concatenate(predictions), np.concatenate(labels))

        return test_acc

    except:
        warnings.warn("Test sets not set in nodes.")
        return None


# -------------------------------------------------------------------------------  BASELINE

def best_accuracy(nodes):
    """ returns both training and testing accuracies """
    best_clf = GradientBoostingClassifier()

    predictions, labels = [], []

    for n in nodes:
        best_clf.fit(n.sample, n.labels)
        predictions.append(best_clf.predict(n.sample))
        labels.append(n.labels)

    best_train_acc = accuracy_score(np.concatenate(predictions), np.concatenate(labels))

    try:
        predictions, labels = [], []
        for n in nodes:
            predictions.append(best_clf.predict(n.test_sample))
            labels.append(n.test_labels)
        
        best_test_acc = accuracy_score(np.concatenate(predictions), np.concatenate(labels))

    except:
        best_test_acc = None

    return best_train_acc, best_test_acc


def random_accuracy(nodes, *args):

    clf = RandomClassifier()

    predictions, labels = [], []

    for n in nodes:
        predictions.append(clf.predict(n.sample))
        labels.append(n.labels)

    train_acc = accuracy_score(np.concatenate(predictions), np.concatenate(labels))

    try:
        predictions, labels = [], []
        for n in nodes:
            predictions.append(clf.predict(n.test_sample))
            labels.append(n.test_labels)
        
        test_acc = accuracy_score(np.concatenate(predictions), np.concatenate(labels))

    except:
        test_acc = None

    return train_acc, test_acc

def maj_class_accuracy(nodes, *args):

    labels = []

    for n in nodes:
        labels += n.labels.tolist()

    plus_points = labels.count(1)
    minus_points = labels.count(-1)

    if plus_points > minus_points:
        train_acc = plus_points / (plus_points + minus_points)
        plus = True
    else:
        train_acc = minus_points / (plus_points + minus_points)
        plus = False

    try:
        labels = []
        for n in nodes:
            labels += n.test_labels.tolist()
        
        if plus:
            test_acc = labels.count(1) / len(labels)
        else:
            test_acc = labels.count(-1) / len(labels)

    except:
        test_acc = None

    return train_acc, test_acc