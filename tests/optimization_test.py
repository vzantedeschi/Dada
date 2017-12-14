import unittest

from copy import deepcopy
import numpy as np
import sys
sys.path.append('./src/')

from evaluation import alpha_variance, clf_variance
from network import complete_graph, centralize_data
from optimization import average_FW
from utils import load_iris_dataset

class Test(unittest.TestCase):

    def setUp(self):
        X, Y = load_iris_dataset()
        self.D = X.shape[1]

        self.nodes = complete_graph(X, Y, nb_nodes=12)
        
    def test_average_FW(self): 
        NB_ITER = 10

        for i in range(NB_ITER):
            with self.subTest(i=i):
                average_FW(self.nodes, self.D, i, callbacks={})
                self.assertEqual(alpha_variance(self.nodes), 0)
                self.assertEqual(clf_variance(self.nodes), 0)


if __name__ == '__main__':
    unittest.main()