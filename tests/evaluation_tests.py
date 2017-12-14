import unittest

import numpy as np

import sys
sys.path.append('./src/')

from evaluation import alpha_variance
from network import complete_graph

class Test(unittest.TestCase):

    def setUp(self):
        self.nodes = complete_graph(np.ones((10, 5)), np.ones((10, 5)), nb_nodes=7)

    def test_alpha_variance(self): 
        # test alphas all equal and null at initialization 
        for n in self.nodes:
            n.init_matrices(5)   
        self.assertEqual(alpha_variance(self.nodes), 0)

        # test null variance 
        for n in self.nodes:
            n.set_alpha(np.eye(5, 1))

        self.assertEqual(alpha_variance(self.nodes), 0)

        # test controlled variance
        for i, n in enumerate(self.nodes):
            n.set_alpha(np.asarray([[1] if j==i else [0] for j in range(5)]))

        self.assertNotEqual(alpha_variance(self.nodes), 0)

if __name__ == '__main__':
    unittest.main()