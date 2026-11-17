"""Train the final diabetes model and save it to model.bin.

Usage:  python train.py
"""
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA_PATH = "data/diabetes.csv"
MODEL_PATH = "model.bin"
RANDOM_STATE = 42

COLUMNS = [
    "pregnancies", "glucose", "blood_pressure", "skin_thickness",
    "insulin", "bmi", "diabetes_pedigree", "age", "outcome",
]
# In this dataset a 0 in these columns really means "not measured"
ZERO_AS_MISSING = ["glucose", "blood_pressure", "skin_thickness", "insulin", "bmi"]


def load_data(path=DATA_PATH):
    df = pd.read_csv(path, header=None, names=COLUMNS)
    df[ZERO_AS_MISSING] = df[ZERO_AS_MISSING].replace(0, np.nan)
    return df


def make_pipeline(model):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", model),
    ])


def get_candidates():
    """Three model families, each with a small parameter grid."""
    return {
        "logistic_regression": (
            make_pipeline(LogisticRegression(max_iter=1000)),
            {"model__C": [0.01, 0.1, 1, 10]},
        ),
        "random_forest": (
            make_pipeline(RandomForestClassifier(random_state=RANDOM_STATE)),
            {"model__n_estimators": [100, 300],
             "model__max_depth": [3, 5, 8],
             "model__min_samples_leaf": [1, 5, 10]},
        ),
        "gradient_boosting": (
            make_pipeline(GradientBoostingClassifier(random_state=RANDOM_STATE)),
            {"model__n_estimators": [50, 100, 200],
             "model__learning_rate": [0.01, 0.05, 0.1],
             "model__max_depth": [2, 3]},
        ),
    }


def main():
    df = load_data()
    X = df.drop(columns="outcome")
    y = df["outcome"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    best_name, best_search = None, None
    for name, (pipe, grid) in get_candidates().items():
        search = GridSearchCV(pipe, grid, cv=cv, scoring="roc_auc", n_jobs=-1)
        search.fit(X_train, y_train)
        print(f"{name:22s} CV AUC = {search.best_score_:.3f}  {search.best_params_}")
        if best_search is None or search.best_score_ > best_search.best_score_:
            best_name, best_search = name, search

    best_model = best_search.best_estimator_
    test_auc = roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1])
    print(f"\nBest model: {best_name} | test AUC = {test_auc:.3f}")

    # Refit the winner on ALL data before saving
    best_model.fit(X, y)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(best_model, f)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
