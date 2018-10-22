from related_works import colearning
from utils import load_school, get_split_per_list

# set graph of nodes with local personalized data

NB_ITER = 10000
random_state = 72018

CV_SPLITS = 3
MU_LIST = [10**i for i in range(-3, 3)]

STEP = 500

X, Y, _, _, adjacency, distances, K, max_nb_instances = load_school(thr=20)
D = X[0].shape[1]

results = {}.fromkeys(MU_LIST, 0.)

for indices in get_split_per_list(X, CV_SPLITS, rnd_state=random_state):

    train_x, test_x, train_y, test_y = [], [], [], []

    for i, inds in enumerate(indices):
        train_x.append(X[i][inds[0]])
        test_x.append(X[i][inds[1]])
        train_y.append(Y[i][inds[0]])
        test_y.append(Y[i][inds[1]])

    for mu in MU_LIST:

        print(mu)
        linear, _ = colearning(K, train_x, train_y, test_x, test_y, D, NB_ITER, adjacency, distances, mu=mu, max_samples_per_node=max_nb_instances, checkevery=10000)

        results[mu] += linear[-1]["test-accuracy"]

print("best mu:", max(results, key=results.get))
