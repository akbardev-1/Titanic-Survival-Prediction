from src.data_loader import load_data
from src.features import build_features
from src.model import train_model
from src.predict import make_submission

def main():
    print("Loading data...")
    train_df, test_df = load_data()

    print("Building features...")
    train_df, test_df, test_ids = build_features(train_df, test_df)

    print("Training model...")
    model = train_model(train_df)

    print("Making submission...")
    make_submission(model, test_df, test_ids)

if __name__ == "__main__":
    main()