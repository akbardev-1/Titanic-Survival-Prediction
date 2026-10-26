import pandas as pd

def build_features(train_df, test_df):
    test_passenger_ids = test_df["PassengerId"].copy()

    for df in [train_df, test_df]:
        # Family size
        df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
        
        # Extract Title
        df["Title"] = df["Name"].str.extract(r' ([A-Za-z]+)\.', expand=False)
        rare_titles = ['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona']
        df["Title"] = df["Title"].replace(rare_titles, 'Rare')
        df["Title"] = df["Title"].replace(['Mlle', 'Ms'], 'Miss')
        df["Title"] = df["Title"].replace('Mme', 'Mrs')
        
        # Drop Name and PassengerId NOW, before combining
        df.drop(columns=["Name", "PassengerId"], inplace=True)

    # One-Hot Encode categorical columns automatically
    combined = pd.concat([train_df.drop('Survived', axis=1), test_df], axis=0)
    combined = pd.get_dummies(combined, drop_first=True)
    
    # Split them back apart
    train_df = combined.iloc[:len(train_df)].copy()
    test_df = combined.iloc[len(train_df):].copy()
    
    # Add Survived back to train_df
    train_df['Survived'] = pd.read_csv("data/train.csv")["Survived"]

    return train_df, test_df, test_passenger_ids