"""
train.py — Train house-price models (Random Forest + XGBoost)
Picks the better performer and saves it as model.pkl
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("⚠  xgboost not installed — will use Random Forest only.\n")


def train():
    # ── Load data ──
    if not os.path.exists("data.csv"):
        print("data.csv not found — generating now...")
        from generate_data import generate_dataset
        df = generate_dataset()
        df.to_csv("data.csv", index=False)
    else:
        df = pd.read_csv("data.csv")

    print(f"Dataset: {len(df)} rows, {df.shape[1]} columns\n")

    FEATURES = ["city", "area", "sqft", "bhk", "bathrooms"]
    TARGET   = "price"

    X = df[FEATURES]
    y = df[TARGET]

    # ── Preprocessing ──
    cat_cols = ["city", "area"]
    num_cols = ["sqft", "bhk", "bathrooms"]

    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ("num", StandardScaler(), num_cols),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ── Candidate models ──
    candidates = {
        "RandomForest": RandomForestRegressor(
            n_estimators=300, max_depth=18, min_samples_leaf=4,
            random_state=42, n_jobs=-1
        ),
    }

    if HAS_XGB:
        candidates["XGBoost"] = XGBRegressor(
            n_estimators=400, max_depth=8, learning_rate=0.08,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbosity=0,
        )

    best_name, best_score, best_pipe = None, -np.inf, None

    for name, model in candidates.items():
        pipe = Pipeline([("pre", preprocessor), ("model", model)])
        cv   = cross_val_score(pipe, X_train, y_train, cv=5,
                               scoring="r2", n_jobs=-1)
        print(f"{name:15s}  CV R² = {cv.mean():.4f} (±{cv.std():.4f})")

        if cv.mean() > best_score:
            best_name, best_score, best_pipe = name, cv.mean(), pipe

    # ── Fit best model on full train set ──
    best_pipe.fit(X_train, y_train)
    y_pred = best_pipe.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2  = r2_score(y_test, y_pred)

    print(f"\nBest model : {best_name}")
    print(f"Test MAE   : ₹{mae:,.0f}")
    print(f"Test R²    : {r2:.4f}")

    # ── Save ──
    with open("model.pkl", "wb") as f:
        pickle.dump(best_pipe, f)
    print("\nSaved → model.pkl")


if __name__ == "__main__":
    train()
