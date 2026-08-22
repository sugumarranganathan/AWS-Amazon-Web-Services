import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import joblib

def append_to_file(filename, *args):
    with open(filename, 'a') as file:
        for value in args:
            file.write(str(value) + '\n')


def train_model():
    # Load dataset from SageMaker input path
    log_file_path = "/opt/ml/output/public/output.txt"
    input_path = os.path.join("/opt/ml/input/data/train", "insurance_pre.csv")
    if not os.path.exists(input_path):
        print(f"Skipping training: file not found at {input_path}")
        return

    df = pd.read_csv(input_path)

    # Split features and label
    X = df.drop("charges", axis=1)
    y = df["charges"]

    # One-hot encode categorical variables and drop the first category
    X = pd.get_dummies(X, drop_first=True)
    print(X.columns)
    append_to_file(log_file_path, "After Get Dummies column names are", X.columns)

    # Handle any missing values
    X = X.fillna(0)
    y = y.fillna(0)

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a random forest regressor
    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    # Predict and evaluate using R²
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print("R-squared score:", r2)
    append_to_file(log_file_path, "R2 value", r2)

    # Save model
    joblib.dump(model, os.path.join("/opt/ml/model", "model.joblib"))


# ✅ Only run this when it's a training job
if __name__ == "__main__":
    train_model()
