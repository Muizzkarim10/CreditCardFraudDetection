import os
import sys
import joblib
import pandas as pd


# =========================
# Paths
# =========================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)


# =========================
# Load artifacts
# =========================

model = joblib.load(
    os.path.join(
        MODEL_DIR,
        "fraud_model.pkl"
    )
)

scaler = joblib.load(
    os.path.join(
        MODEL_DIR,
        "scaler.pkl"
    )
)

threshold = joblib.load(
    os.path.join(
        MODEL_DIR,
        "threshold.pkl"
    )
)


# =========================
# Feature names
# =========================

FEATURES = [
    "Time",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
    "V7",
    "V8",
    "V9",
    "V10",
    "V11",
    "V12",
    "V13",
    "V14",
    "V15",
    "V16",
    "V17",
    "V18",
    "V19",
    "V20",
    "V21",
    "V22",
    "V23",
    "V24",
    "V25",
    "V26",
    "V27",
    "V28",
    "Amount"
]


# =========================
# Prediction function
# =========================

def predict_transactions(df):

    missing_features = [
        feature
        for feature in FEATURES
        if feature not in df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing features: {missing_features}"
        )

    X = df[FEATURES]

    # Apply the same scaler used during training
    X_scaled = scaler.transform(X)

    # Get fraud probabilities
    probabilities = model.predict_proba(
        X_scaled
    )[:, 1]

    # Apply saved threshold
    predictions = (
        probabilities >= threshold
    ).astype(int)

    results = df.copy()

    results["Fraud_Probability"] = probabilities
    results["Prediction"] = predictions

    results["Prediction"] = results[
        "Prediction"
    ].map({
        0: "LEGITIMATE",
        1: "FRAUD"
    })

    return results


# =========================
# Main
# =========================

if __name__ == "__main__":

    if len(sys.argv) != 2:

        print(
            "\nUsage:"
        )

        print(
            "python src/predict.py "
            "<csv_file>"
        )

        print(
            "\nExample:"
        )

        print(
            "python src/predict.py "
            "data/test_transactions.csv"
        )

        sys.exit(1)

    input_path = sys.argv[1]

    if not os.path.exists(input_path):

        print(
            f"\nFile not found: {input_path}"
        )

        sys.exit(1)

    print(
        f"\nLoading transactions from: "
        f"{input_path}"
    )

    df = pd.read_csv(input_path)

    results = predict_transactions(df)

    print(
        f"\nProcessed {len(results)} "
        f"transactions."
    )

    print(
        f"Fraud threshold: {threshold:.3f}"
    )

    print("\nPrediction summary:")

    print(
        results["Prediction"].value_counts()
    )

    print("\nFirst predictions:")

    print(
        results[
            [
                "Fraud_Probability",
                "Prediction"
            ]
        ].head(10)
    )

    output_path = os.path.join(
        os.path.dirname(input_path),
        "predictions.csv"
    )

    results.to_csv(
        output_path,
        index=False
    )

    print(
        f"\nPredictions saved to: "
        f"{output_path}"
    )