# Titanic Survival Prediction 🚢

This project uses Machine Learning to predict whether a passenger survived the Titanic disaster based on features like age, sex, and passenger class.

## Approach
1. **Data Cleaning**: Filled missing values for `Age`, `Fare`, and `Embarked`.
2. **Feature Engineering**: 
   - Extracted passenger `Title` (Mr, Mrs, Miss, etc.) from the `Name` column.
   - Created a new `FamilySize` feature by combining `SibSp` and `Parch`.
3. **Preprocessing**: Used One-Hot Encoding to convert categorical text data (Sex, Embarked, Title) into numbers.
4. **Model**: Trained a Random Forest Classifier to make the predictions.

## Results
- **Validation Accuracy**: ~82.7%

## How to Run
1. Install the required libraries:
   ```bash
   pip install pandas scikit-learn