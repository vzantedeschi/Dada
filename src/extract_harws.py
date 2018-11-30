import pandas as pd
from sklearn.model_selection import train_test_split

#dataset from https://www.kaggle.com/uciml/human-activity-recognition-with-smartphones/home

LABELS = {
    "WALKING": 1,
    "WALKING_UPSTAIRS": 1, 
    "WALKING_DOWNSTAIRS": 1, 
    "STANDING": -1,
    "SITTING": -1, 
    "LAYING": -1
}

K = 30
rnd_state = 13112018

train_df = pd.read_csv('datasets/human-activity-recognition-with-smartphones/train.csv')
test_df = pd.read_csv('datasets/human-activity-recognition-with-smartphones/test.csv')

df = pd.concat([train_df, test_df])
assert df['subject'].nunique() == K

# df.sort_values(["subject"], inplace=True)

print("nb instances= {}, nb features= {}".format(df.shape[0], df.shape[1]-2))

# count by subject
print("\ninstances per agent")
print(df['subject'].value_counts())

# count by activity
print("\ninstances per task")
print(df['Activity'].value_counts())

# binarize problem
df.Activity.replace(LABELS, inplace=True)
print(df['Activity'].value_counts())

#create train and test
train, test = train_test_split(df, train_size=0.02, random_state=rnd_state)

print("\nTRAIN")
print("nb instances= {}, nb features= {}".format(train.shape[0], train.shape[1]-2))

# count by activity
print("\ninstances per task")
print(train['Activity'].value_counts())

assert train['subject'].nunique() == K

print("\nTEST")
print("nb instances= {}, nb features= {}".format(test.shape[0], test.shape[1]-2))

# count by acitivity
print("\ninstances per task")
print(test['Activity'].value_counts())

assert test['subject'].nunique() == K

#save splits
train.to_csv("datasets/harws_train.csv")
test.to_csv("datasets/harws_test.csv")

### second problem: WALKING_UPSTAIRS vs walking WALKING_DOWNSTAIRS
df = pd.concat([train_df, test_df])
df = df[(df["Activity"] == "WALKING_DOWNSTAIRS") | (df["Activity"] == "WALKING_UPSTAIRS")]

print("nb instances= {}, nb features= {}".format(df.shape[0], df.shape[1]-2))

# binarize problem
df.Activity.replace(["WALKING_DOWNSTAIRS", "WALKING_UPSTAIRS"], [-1, 1], inplace=True)
print(df['Activity'].value_counts())

#create train and test
train, test = train_test_split(df, train_size=0.05, random_state=rnd_state)

print("\nTRAIN")
print("nb instances= {}, nb features= {}".format(train.shape[0], train.shape[1]-2))

# count by activity
print("\ninstances per task")
print(train['Activity'].value_counts())

assert train['subject'].nunique() == K

print("\nTEST")
print("nb instances= {}, nb features= {}".format(test.shape[0], test.shape[1]-2))

# count by acitivity
print("\ninstances per task")
print(test['Activity'].value_counts())

assert test['subject'].nunique() == K

#save splits
train.to_csv("datasets/harws_train_walking.csv")
test.to_csv("datasets/harws_test_walking.csv")