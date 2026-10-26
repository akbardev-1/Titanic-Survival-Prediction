import pandas as pd
import os

def make_submission(model, test_df, test_passenger_ids, out_path="output/submission.csv"):
    os.makedirs("output", exist_ok=True)
    
    predictions = model.predict(test_df)

    submission = pd.DataFrame({
        "PassengerId": test_passenger_ids,
        "Survived": predictions
    })

    submission.to_csv(out_path, index=False)
    print(f"Submission saved to {out_path}")