import unittest

from copy import deepcopy

import sys
sys.path.append('./src/')

from evaluation import mean_accuracy, loss
from network import line_network, complete_graph
from optimization import centralized_FW
from utils import load_iris_dataset, load_breast_dataset


class TestCentralized(unittest.TestCase):

    def test_iris(self):
        X, Y = load_iris_dataset()
        D = X.shape[1]
        NB_ITER = 10

        # set graph
        nodes = line_network(X, Y, nb_nodes=1)
        self.assertEqual(nodes[0].sample.shape[1], 5)

        results = centralized_FW(nodes, D, nb_iter=NB_ITER, callbacks={"mean_accuracy":[mean_accuracy, []], "loss": [loss, []]})
        self.assertEqual(nodes[0].clf.shape, (1, 5))
        self.assertEqual(len(results), NB_ITER)

        acc = mean_accuracy(nodes)
        self.assertGreaterEqual(acc[0], 0.95)

    def test_breast(self):
        X, Y = load_breast_dataset()
        D = X.shape[1]
        # set graph
        nodes = complete_graph(X, Y, nb_nodes=1)

        NB_ITER = 10
        results = centralized_FW(nodes, D, nb_iter=NB_ITER, callbacks={"mean_accuracy":[mean_accuracy, []]})

        self.assertGreaterEqual(results[9]["mean_accuracy"][0], results[0]["mean_accuracy"][0])

if __name__ == '__main__':
    unittest.main()