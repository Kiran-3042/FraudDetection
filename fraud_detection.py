import pandas as pd
import seaborn as sns
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import precision_score,recall_score,f1_score,confusion_matrix
creditcard_df=pd.read_csv("creditcard.csv")

print(creditcard_df.head())
print(creditcard_df.shape)

print(creditcard_df.isnull().sum())
print(creditcard_df.info())

print(creditcard_df["Class"].value_counts())
sns.histplot(data=creditcard_df,x="Amount",bins=5)

x=creditcard_df.drop("Class",axis=1)
y=creditcard_df["Class"]

print(x.shape)
print(y.shape)

x_train,x_test,y_train,y_test=train_test_split(
    x,y,test_size=0.2,random_state=42,stratify=y
)
print(x_train.shape)

print(y_train.shape)
print(x_test.shape)

print(y_test.shape)
scaler_x=StandardScaler().fit(x_train)

x_train=scaler_x.transform(x_train)
x_test=scaler_x.transform(x_test)

model=LogisticRegression()
model.fit(x_train,y_train)

prediction=model.predict(x_test)
original_cm=confusion_matrix(y_test,prediction)
original_precision=precision_score(y_test,prediction)

original_recall=recall_score(y_test,prediction)
original_f1=f1_score(y_test,prediction)

model=LogisticRegression(class_weight="balanced")

model.fit(x_train,y_train)
predictions=model.predict(x_test)
balanced_cm=confusion_matrix(y_test,predictions)
balanced_precision=precision_score(y_test,predictions)
balanced_recall=recall_score(y_test,predictions)
balanced_f1=f1_score(y_test,predictions)

f1=f1_score(y_test,predictions)
precision=precision_score(y_test,predictions)

recall=recall_score(y_test,predictions)
cm=confusion_matrix(y_test,predictions)

probability=model.predict_proba(x_test)
fraud_probability=probability[:,1]

threshold_predictions=(fraud_probability>=1).astype(int)
threshold_f1=f1_score(y_test,threshold_predictions)

threshold_precision=precision_score(y_test,threshold_predictions)
threshold_recall=recall_score(y_test,threshold_predictions)

threshold_cm=confusion_matrix(y_test,threshold_predictions)
joblib.dump(model,"fraud_model.pkl")
joblib.dump(scaler_x,"scaler.pkl")
print("F1 score=",f1)
print("Precision=",precision)

print("Recall=",recall)
print("Confusion Matrix=",cm)

print("Probability=",probability)
print("Fraud Probability=",fraud_probability)

print("Threshold Predictions=",threshold_predictions)
print("Threshold_f1=",threshold_f1)

print("Threshold_precision=",threshold_precision)
print("Threshold_recall=",threshold_recall)
print()
print("Original Model Precision=",original_precision)
print("Original Model Recall=",original_recall)
print("Original Model F1=",original_f1)
print()
print("Balanced Model Precision=",balanced_precision)
print("Balanced Model Recall=",balanced_recall)
print("Balanced Model F1=",balanced_f1)
print()
print("Original Model Confusion Matrix=",original_cm)
print("Balanced Model Confusion Matrix=",balanced_cm)

print("Saving files to:",os.getcwd())