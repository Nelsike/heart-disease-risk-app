"""
Train the random forest classifier and save model + feature names.

This script reproduces the model used by app.py. Run it once to regenerate
heart_rf_model.joblib and feature_names.joblib if either is missing or if
the dataset changes.

Usage:
    python train.py
"""

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

DATA_PATH = "Heart_disease_cleveland_new.csv"
MODEL_PATH = "heart_rf_model.joblib"
FEATURES_PATH = "feature_names.joblib"
RANDOM_STATE = 42


def main():
    df = pd.read_csv(DATA_PATH)
    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    model = RandomForestClassifier(random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    print(f"Test accuracy: {accuracy_score(y_test, pred):.4f}")
    print(f"Test ROC AUC:  {roc_auc_score(y_test, proba):.4f}")

    joblib.dump(model, MODEL_PATH)
    joblib.dump(list(X.columns), FEATURES_PATH)
    print(f"Saved {MODEL_PATH} and {FEATURES_PATH}")


if __name__ == "__main__":
    main()
