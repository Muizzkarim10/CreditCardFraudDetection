# 💳 Credit Card Fraud Detection

A machine learning project for detecting fraudulent credit card transactions using **Logistic Regression, StandardScaler, and SMOTE**.

## 📌 Overview

The dataset contains **284,807 transactions**, with only **492 fraudulent transactions (~0.17%)**, making this a highly imbalanced classification problem.

The project focuses on handling class imbalance and optimizing the classification threshold to improve fraud detection.

## 🔍 Dataset

Features:

- `Time`
- `Amount`
- `V1` – `V28` (PCA-transformed features)
- `Class` — target variable

```text
0 → Legitimate
1 → Fraud
```

No missing values or duplicate transactions were found.

## 🤖 Approach

1. Train/test split with stratification
2. Train/validation split
3. Feature scaling using `StandardScaler`
4. Class balancing using **SMOTE**
5. Logistic Regression
6. Threshold tuning using the validation set
7. Final evaluation on the untouched test set

The final classification threshold was **0.999** rather than the default `0.5`.

## 📊 Results

Final test performance:

| Metric | Score |
|---|---:|
| ROC-AUC | **0.974** |
| Fraud Precision | **61%** |
| Fraud Recall | **85%** |
| Fraud F1 | **71%** |

### Confusion Matrix

```text
[[56810    54]
 [   15    83]]
```

The model detected **83 of 98 fraud cases** in the test set.

## 🛠️ Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- imbalanced-learn
- Matplotlib
- Seaborn
- Joblib
- Jupyter Notebook
- Git & Git LFS

## 📁 Project Structure

```text
CreditCardFraudDetection/
├── data/
│   └── creditcard.csv
├── models/
│   ├── fraud_model.pkl
│   ├── scaler.pkl
│   └── threshold.pkl
├── notebooks/
│   └── eda.ipynb
├── src/
│   ├── train.py
│   └── predict.py
├── .gitignore
├── requirements.txt
└── ReadMe.md
```

## 🚀 Usage

Install dependencies:

```bash
pip install -r requirements.txt
```

Train the model:

```bash
python src/train.py
```

Run predictions on a CSV:

```bash
python src/predict.py data/test_transactions.csv
```

Predictions are saved to:

```text
data/predictions.csv
```

## 🧠 Key Concepts

- Imbalanced classification
- SMOTE
- Feature scaling
- Data leakage prevention
- Precision vs recall
- Threshold tuning
- ROC-AUC
- Model persistence
- Batch inference

## 👤 Author

**Muizz Karim**