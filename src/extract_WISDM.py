from scipy.io import arff

def get_users(data):

    ids = set()

    for d in data:

        if len(d) == 45:
            ids.add(int(d[0]))

    return ids 

def get_data_by_user(finput, users):

    data = []

    for line in finput:

        slices = line.split(',')
        slices = list(filter(None, slices))

        if len(slices) < 6:
            continue

        if int(slices[0]) in users:
            data.append(slices)

    return data


with open("datasets/WISDM_users.txt", "r") as users_file:

    labeled_data, meta = arff.loadarff("datasets/WISDM_labeled.arff")

    user_ids = get_users(labeled_data)
    print("nb users in labeled data", len(user_ids))

    selected_users = get_data_by_user(users_file, user_ids)
    print("nb valid users in labeled data", len(selected_users))

with open("datasets/WISDM_users.txt", "r") as users_file:

    unlabeled_data, _ = arff.loadarff("datasets/WISDM_unlabeled.arff")

    user_ids = get_users(unlabeled_data)
    print("nb users in unlabeled data", len(user_ids))
    
    selected_users = get_data_by_user(users_file, user_ids)
    print("nb valid users in unlabeled data", len(selected_users))