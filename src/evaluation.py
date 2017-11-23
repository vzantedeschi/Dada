import numpy as np

from sklearn.metrics import accuracy_score

def alpha_variance(nodes, *args):
    return np.mean(np.var([n.alpha for n in nodes], axis=0))

def mean_accuracy(nodes, *args):
    """ returns mean train accuracy, mean test accuracy
    """
    train_acc, test_acc = [], []
    for n in nodes:
        train_acc.append(accuracy_score(n.predict(n.sample), n.labels))
        test_acc.append(accuracy_score(n.predict(n.test_sample), n.test_labels))

    return np.mean(train_acc), np.mean(test_acc)