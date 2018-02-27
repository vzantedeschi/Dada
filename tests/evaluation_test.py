import unittest

from copy import deepcopy
import numpy as np
import sys
sys.path.append('./src/')

from classification import get_double_basis
from evaluation import alpha_variance, loss, central_loss, mean_accuracy, central_accuracy
from optimization import average_FW, centralized_FW
from network import complete_graph, centralize_data
from utils import load_iris_dataset

class TestFakeData(unittest.TestCase):

    def setUp(self):
        self.nodes = complete_graph(np.ones((10, 5)), np.ones(10), nb_nodes=7)
        base_clfs = get_double_basis(6, 6)
        for n in self.nodes:
            n.init_matrices(base_clfs)

    def test_alpha_variance(self): 
        # test alphas all equal and null at initialization    
        self.assertEqual(alpha_variance(self.nodes), 0)

        # test null variance 
        for n in self.nodes:
            n.set_alpha(np.eye(6, 1))

        self.assertEqual(alpha_variance(self.nodes), 0)

        # test controlled variance
        for i, n in enumerate(self.nodes):
            n.set_alpha(np.asarray([[1] if j==i else [0] for j in range(5)]))

        self.assertNotEqual(alpha_variance(self.nodes), 0)

    def test_loss(self):

        self.assertEqual(loss(self.nodes), 0)
        self.assertEqual(central_loss(self.nodes), 0)

        node = centralize_data(self.nodes)

        base_clfs = get_double_basis(6, 6)
        for n in self.nodes:
            n.init_matrices(base_clfs)
            n.set_alpha(np.eye(6, 1))

        node.init_matrices(base_clfs)
        node.set_alpha(np.eye(6, 1))

        loss1 = central_loss(self.nodes)
        loss2 = loss([node])
        self.assertEqual(loss1, loss2)

    def test_accuracy(self): 
        # test null accuracy at begging    
        self.assertEqual(mean_accuracy(self.nodes)[0], 0)

        node = centralize_data(self.nodes)
        base_clfs = get_double_basis(6, 6)
        for n in self.nodes:
            n.init_matrices(base_clfs)
            n.set_alpha(np.eye(6, 1))

        node.init_matrices(base_clfs)
        node.set_alpha(np.eye(6, 1))

        acc1 = central_accuracy(self.nodes)[0]
        acc2 = mean_accuracy([node])[0]
        self.assertEqual(acc1, acc2)

class TestIris(unittest.TestCase):

    def setUp(self):
        X, Y = load_iris_dataset()
        self.D = X.shape[1]

        self.nodes = complete_graph(X, Y, nb_nodes=12) 

    def test_accuracy(self): 

        base_clfs = get_double_basis(self.D, self.D+1)
        average_FW(self.nodes, base_clfs, nb_iter=10, callbacks={})
        acc1 = central_accuracy(self.nodes)[0]

        centralized_FW(self.nodes, base_clfs, nb_iter=10, callbacks={})
        acc2 = central_accuracy(self.nodes)[0]
        acc3 = mean_accuracy(self.nodes)[0]
        self.assertAlmostEqual(acc2, acc3) 
        self.assertAlmostEqual(acc1, acc2, places=1)


if __name__ == '__main__':
    unittest.main()