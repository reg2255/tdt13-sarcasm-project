import pandas as pd
from sklearn.model_selection import train_test_split

def load_train(path):
    train_data = pd.read_csv(path, index_col=0)

    # remove tweet with missing text
    train_data = train_data.dropna(subset="tweet")

    # remove duplicate tweets
    # conflicting labels: remove all
    labels_per_tweet = train_data.groupby("tweet")["sarcastic"].nunique()
    conflicting_tweets = labels_per_tweet[labels_per_tweet > 1].index
    train_data = train_data[~train_data["tweet"].isin(conflicting_tweets)]
    # not conflicting labels: keep one of them
    train_data = train_data.drop_duplicates(subset="tweet")

    return train_data

def load_test(path_a, path_b):
    test_data_A = pd.read_csv(path_a)
    test_data_B = pd.read_csv(path_b)
    # can look at rename() if I want to rename the text column to tweet as in training set

    return test_data_A, test_data_B

def split_train_val(data, val_size=0.15, seed=42):
    # split training data into training and validation
    train_data, val_data = train_test_split(data, stratify=data["sarcastic"], random_state=seed, test_size=val_size)
    return train_data, val_data

if __name__ == "__main__":
    data = load_train("./data/train.En.csv")
    print(f"After cleaning: {len(data)} tweets")
    print(data["sarcastic"].value_counts())

    train_data, val_data = split_train_val(data)
    print(f"Train: {len(train_data)}, validation: {len(val_data)}")
    print(train_data["sarcastic"].value_counts(normalize=True))
    print(val_data["sarcastic"].value_counts(normalize=True))