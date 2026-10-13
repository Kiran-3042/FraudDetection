# 💳 Credit Card Fraud Detection

A Machine Learning project that detects potentially fraudulent credit card transactions using Logistic Regression, Scikit-learn, and Streamlit.

## 📌 Project Overview

Credit card fraud detection is an imbalanced classification problem because fraudulent transactions are much rarer than legitimate transactions.

This project trains and evaluates Logistic Regression models and provides an interactive dashboard for transaction predictions.

## 🚀 Features

- Exploratory Data Analysis (EDA)
- Data preprocessing and feature scaling
- Train-test splitting with stratification
- Logistic Regression classification
- Class imbalance handling using class weights
- Model evaluation using precision, recall, F1-score, and confusion matrix
- Fraud probability estimation
- Interactive Streamlit dashboard
- Saved model and scaler for reuse

## 🛠️ Technologies Used

- Python
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit
- Joblib

## 📂 Project Structure

```text
FraudDetection/
├── app.py
├── fraud_detection.py
├── main.py
├── fraud_model.pkl
├── scaler.pkl
├── requirements.txt
└── README.md
```

The dataset, `creditcard.csv`, is not included in this repository.

## 📊 Dataset

The project uses a credit card transaction dataset containing 284,807 transactions and 31 columns.

- `Class = 0`: Legitimate transaction
- `Class = 1`: Fraudulent transaction
- `V1`–`V28`: Anonymized features
- `Amount`: Transaction amount
- `Time`: Transaction time

The dataset contains 492 fraudulent transactions, making class imbalance an important consideration.

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/Kiran-3042/FraudDetection.git
cd FraudDetection
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

Install the packages used by the project:

```bash
pip install pandas scikit-learn matplotlib seaborn streamlit joblib
```

### 4. Add the dataset

Place `creditcard.csv` in the project root directory.

### 5. Run the dashboard

```bash
streamlit run app.py
```

## 📈 Model Evaluation

The project compares a standard Logistic Regression model with a class-weight-balanced Logistic Regression model.

The evaluation includes:

- Precision
- Recall
- F1-score
- Confusion matrix

Because missing a fraudulent transaction can be costly, recall is an important metric. However, precision must also be considered to reduce false alarms.

## ⚠️ Disclaimer

This project is intended for educational purposes. Predictions are estimates and should not be used as the sole basis for real financial decisions.
