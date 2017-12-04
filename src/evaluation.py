import numpy as np

from sklearn.metrics import accuracy_score

def alpha_variance(nodes, *args):
    return np.mean(np.var([n.alpha for n in nodes], axis=0))

def loss(nodes, *args):
    return sum([np.sum(n.compute_weights(distr=False)) for n in nodes])

def mean_accuracy(nodes, *args):
    """ returns mean train accuracy, mean test accuracy
    """
    train_acc = []
    for n in nodes:
        train_acc.append(accuracy_score(n.predict(n.sample), n.labels))
    mean_train_acc = np.mean(train_acc)

    try:
        test_acc = []
        for n in nodes:
            test_acc.append(accuracy_score(n.predict(n.test_sample), n.test_labels))
        mean_test_acc = np.mean(test_acc)
    except:
        mean_test_acc = None

    return mean_train_acc, mean_test_acc