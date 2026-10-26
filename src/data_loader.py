import pandas as pd

def load_data(train_path="data/train.csv", test_path="data/test.csv"):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Fill missing values
    train_df["Age"] = train_df["Age"].fillna(train_df["Age"].median())
    test_df["Age"] = test_df["Age"].fillna(test_df["Age"].median())
    test_df["Fare"] = test_df["Fare"].fillna(test_df["Fare"].median())
    train_df["Embarked"] = train_df["Embarked"].fillna(train_df["Embarked"].mode()[0])

    # Drop columns we don't need
    drop_cols = ["Cabin", "Ticket"]
    train_df = train_df.drop(columns=drop_cols)
    test_df = test_df.drop(columns=drop_cols)

    return train_df, test_df