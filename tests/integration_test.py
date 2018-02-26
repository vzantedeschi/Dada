import unittest

from copy import deepcopy

import sys
sys.path.append('./src/')

from classification import get_stumps
from evaluation import mean_accuracy, loss
from network import line_network, complete_graph
from optimization import centralized_FW
from utils import load_iris_dataset, load_breast_dataset

def toy_dataset(m, seed=0, t="xor", d=2):
    import numpy as np
    np.random.seed(seed)

    if t=="xor":
        X = np.random.uniform(low=tuple([-1.]*d), high=tuple([1.]*d), size=(m,d))
        Y = np.asarray([2*int(v>=0)-1 for v in X[:,0]*X[:,1]])

    elif t=="normal":
        m2 = int(m/2)
        X1 = np.random.normal((0,-2),(1,2),(m2,d))
        Y1 = np.ones((m2,1))

        X2 = np.random.normal((0,2),(1,2),(m2,d))
        Y2 = -np.ones((m2,1))

        X = np.r_[X1,X2]
        Y = np.r_[Y1,Y2]

    elif t=="swissroll":
        m2 = int(m/2)
        X1,_ = make_swiss_roll(n_samples=m2,noise=0)
        Y1 = np.ones((m2,1))

        X2 = np.random.uniform(low=tuple([-1.]*d),high=tuple([1.]*d),size=(m2,d))
        Y2 = -np.ones((m2,1))

        X = np.r_[X1[:,::2]/15,X2]
        Y = np.r_[Y1,Y2]

    assert X.shape == (m,d)
    assert Y.shape == (m,)

    return X, Y

class TestCentralized(unittest.TestCase):

    def test_iris(self):
        X, Y = load_iris_dataset()
        D = X.shape[1]
        NB_ITER = 100

        # set graph
        nodes = line_network(X, Y, nb_nodes=1)
        self.assertEqual(nodes[0].sample.shape[1], 5)

        results = centralized_FW(nodes, 2*D, nb_iter=NB_ITER, callbacks={"mean_accuracy":[mean_accuracy, []], "loss": [loss, []]})
        self.assertEqual(nodes[0].alpha.shape, (2*D, 1))
        self.assertEqual(len(results), NB_ITER+1)

        acc = mean_accuracy(nodes)
        self.assertGreaterEqual(acc[0], 0.95)

    def test_breast(self):
        X, Y = load_breast_dataset()
        D = X.shape[1]
        # set graph
        nodes = complete_graph(X, Y, nb_nodes=1)

        NB_ITER = 100
        results = centralized_FW(nodes, 2*D, nb_iter=NB_ITER, callbacks={"mean_accuracy":[mean_accuracy, []]})

        self.assertGreaterEqual(results[9]["mean_accuracy"][0], results[0]["mean_accuracy"][0])

        acc = mean_accuracy(nodes)
        self.assertGreaterEqual(acc[0], 0.60)


    # def test_xor(self):

    #     X, Y = toy_dataset(100)
    #     D = X.shape[1]
    #     # set graph
    #     nodes = complete_graph(X, Y, nb_nodes=1)

    #     NB_ITER = 100
    #     results = centralized_FW(nodes, 2*D, get_stumps, nb_iter=NB_ITER, callbacks={"mean_accuracy":[mean_accuracy, []]})

    #     self.assertGreaterEqual(results[9]["mean_accuracy"][0], results[0]["mean_accuracy"][0])

    #     acc = mean_accuracy(nodes)
    #     self.assertEqual(acc[0], 1.)

if __name__ == '__main__':
    unittest.main()