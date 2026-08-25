import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from imblearn.over_sampling import SMOTE


# =========================
# Paths
# =========================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "creditcard.csv"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

os.makedirs(MODEL_DIR, exist_ok=True)


# =========================
# Load data
# =========================

print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {df.shape}")


# =========================
# Prepare data
# =========================

X = df.drop("Class", axis=1)
y = df["Class"]


# =========================
# Train / Test split
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# =========================
# Train / Validation split
# =========================

X_train_sub, X_val, y_train_sub, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.2,
    random_state=42,
    stratify=y_train
)


# =========================
# Feature scaling
# =========================

print("Scaling features...")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train_sub)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)


# =========================
# SMOTE
# =========================

print("Applying SMOTE...")

smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train_scaled,
    y_train_sub
)

print("\nClass distribution after SMOTE:")
print(y_train_smote.value_counts())


# =========================
# Train Logistic Regression
# =========================

print("\nTraining Logistic Regression...")

model = LogisticRegression(
    max_iter=1000
)

model.fit(
    X_train_smote,
    y_train_smote
)


# =========================
# Find best threshold
# =========================

print("\nFinding optimal threshold...")

y_val_prob = model.predict_proba(
    X_val_scaled
)[:, 1]

thresholds = np.arange(
    0.90,
    1.001,
    0.001
)

results = []

for threshold in thresholds:

    y_val_pred = (
        y_val_prob >= threshold
    ).astype(int)

    results.append({
        "threshold": threshold,
        "precision": precision_score(
            y_val,
            y_val_pred,
            zero_division=0
        ),
        "recall": recall_score(
            y_val,
            y_val_pred
        ),
        "f1": f1_score(
            y_val,
            y_val_pred
        )
    })


results_df = pd.DataFrame(results)

best_row = results_df.loc[
    results_df["f1"].idxmax()
]

final_threshold = float(
    best_row["threshold"]
)

print("\nBest validation threshold:")
print(f"Threshold: {final_threshold:.3f}")
print(f"Precision: {best_row['precision']:.3f}")
print(f"Recall:    {best_row['recall']:.3f}")
print(f"F1:        {best_row['f1']:.3f}")


# =========================
# Final test evaluation
# =========================

print("\nEvaluating on test set...")

y_test_prob = model.predict_proba(
    X_test_scaled
)[:, 1]

y_test_pred = (
    y_test_prob >= final_threshold
).astype(int)


print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_test_pred))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_test_pred
    )
)

print(
    f"ROC-AUC: {roc_auc_score(y_test, y_test_prob):.4f}"
)


# =========================
# Save artifacts
# =========================

print("\nSaving model artifacts...")

joblib.dump(
    model,
    os.path.join(
        MODEL_DIR,
        "fraud_model.pkl"
    )
)

joblib.dump(
    scaler,
    os.path.join(
        MODEL_DIR,
        "scaler.pkl"
    )
)

joblib.dump(
    final_threshold,
    os.path.join(
        MODEL_DIR,
        "threshold.pkl"
    )
)

print("\nDone.")
print(f"Model saved to: {MODEL_DIR}")