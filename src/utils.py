import numpy as np

from sklearn.datasets import load_wine
from sklearn.preprocessing import normalize, scale

# ---------------------------------------------------------------------------------- LOAD DATASETS

def load_wine_dataset():
    
    X, Y = load_wine(return_X_y=True)

    # keep only two classes with labels -1,1
    indices = Y != 2
    Y, X = Y[indices], X[indices]
    Y[Y==0] = -1

    return scale(X), Y

# def load_dense_dataset(dataset_name):
#     dataset = np.loadtxt(DATAPATH + dataset_name + ".txt")
#     if dataset_name == "sonar":
#         x, y = np.split(dataset, [-1], axis=1)
#     elif dataset_name == "ionosphere":
#         x, y = np.split(dataset, [-1], axis=1)
#     elif dataset_name == "heart-statlog":
#         y, x = np.split(dataset, [1], axis=1)

#     else:
#         raise Exception("Unknown dataset: please implement a loader.")

#     return scale(x), y