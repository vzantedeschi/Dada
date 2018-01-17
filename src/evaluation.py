from math import log
import numpy as np

from sklearn.metrics import accuracy_score

def alpha_variance(nodes, *args):
    return np.around(np.mean(np.var([n.alpha for n in nodes], axis=0)), decimals=10)

def clf_variance(nodes, *args):
    return np.around(np.mean(np.var([n.clf for n in nodes], axis=0)), decimals=10)

def loss(nodes, *args):
    return np.sum([log(np.mean(n.compute_weights(distr=False))) for n in nodes])

def central_loss(nodes, *args):
    return log(np.mean(np.concatenate([n.compute_weights(distr=False) for n in nodes])))

def accuracies(nodes, *args):
    """ returns train accuracies, test accuracies
    """
    train_acc = []
    for n in nodes:
        train_acc.append(accuracy_score(n.predict(n.sample), n.labels))

    test_acc = []
    try:

        for n in nodes:
            test_acc.append(accuracy_score(n.predict(n.test_sample), n.test_labels))

    except:
        pass

    return train_acc, test_acc

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

def central_accuracy(nodes, *args):
    """ returns train accuracy, test accuracy
    """

    predictions, labels = [], []
    for n in nodes:
        predictions.append(n.predict(n.sample))
        labels.append(n.labels)
    
    train_acc = accuracy_score(np.concatenate(predictions), np.concatenate(labels))

    try:
        predictions, labels = [], []
        for n in nodes:
            predictions.append(n.predict(n.test_sample))
            labels.append(n.test_labels)
        
        test_acc = accuracy_score(np.concatenate(predictions), np.concatenate(labels))

    except:
        test_acc = None

    return train_acc, test_acc