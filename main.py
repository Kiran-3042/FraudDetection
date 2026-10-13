import joblib
import pandas as pd



model=joblib.load("fraud_model.pkl")
scaler_x=joblib.load("scaler.pkl")

creditcard_df=pd.read_csv("creditcard.csv")

x=creditcard_df.drop("Class",axis=1)
y=creditcard_df["Class"]

sample=x.iloc[[0,1,2]]
actual=y.iloc[[0,1,2]]

sample_scaled=scaler_x.transform(sample)

predictions=model.predict(sample_scaled)

print("Actual classes:",actual.tolist())
print("Predicted class:",predictions.tolist())
