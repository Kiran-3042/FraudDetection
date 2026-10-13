import streamlit as st
import pandas as pd
import joblib

from fraud_detection import fraud_probability
from main import scaler_x

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

model=joblib.load("fraud_model.pkl")
scaler_x=joblib.load("scaler.pkl")
creditcard_df=pd.read_csv("creditcard.csv")

THRESHOLD=0.95

st.title("💳 Credit Card Fraud Detection")
st.write("Machine Learning Dashboard")
st.divider()

total_transactions=len(creditcard_df)
legitimate_count=int((creditcard_df["Class"]==0).sum())
fraud_count=int((creditcard_df["Class"]==1).sum())
fraud_rate=(fraud_count/total_transactions)*100

col1, col2,col3,col4=st.columns(4)

col1.metric("Total Transactions",f"{total_transactions:,}")
col2.metric("Legitimate",f"{legitimate_count:,}")
col3.metric("Fraudulent",f"{fraud_count:,}")
col4.metric("Fraud Rate",f"{fraud_rate:.2f}")

st.divider()

st.subheader("🔍 Transaction Prediction")

transaction_id=st.selectbox(
    "Select a transaction",
    range(len(creditcard_df)),
    format_func=lambda x: f"Transaction {x+1}"
)

selected_transaction=creditcard_df.iloc[transaction_id]

col1,col2=st.columns(2)

col1.metric("Transaction Amount",f"{selected_transaction["Amount"]:.2f}")
col2.metric("Transaction Time",f"{selected_transaction["Time"]:.0f}")

if st.button("Predict Transaction",type="primary"):
    sample=creditcard_df.drop(
        columns=["Class"]
    ).iloc[[transaction_id]]

    sample_scaled=scaler_x.transform(sample)

    fraud_probability=model.predict_proba(sample_scaled)[0][1]

    st.subheader("Prediction Result")

    if fraud_probability>=THRESHOLD:
        st.error("🚨 Potential Fraud Detected")
    else:
        st.success("✅ Not Flagged as fraud at a selected Threshold")

    st.metric(
        "Estimated Fraud Probability",
        f"{fraud_probability:.2%}"
    )

    st.progress(float(fraud_probability))
    st.caption(f"Classification threshold:{THRESHOLD:.0%}")

st.divider()
st.subheader("📊 Transaction Class Distribution")

class_counts=(
    creditcard_df["Class"]
    .value_counts()
    .reindex([0,1],fill_value=0)
)

chart_data=pd.DataFrame({
    "Transaction Type":["Legitimate","Fraudulent"],
    "Count":class_counts.values
}).set_index("Transaction Type")

st.bar_chart(chart_data)

st.caption(
    "Educational project using a historical dataset."
    "Predictions are model estimates, not guarantees"
)