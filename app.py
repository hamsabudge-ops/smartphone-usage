import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
print(df.isnull().sum())
sns.countplot(x='gender', data=df)
plt.title("Gender Distribution")
plt.show()
sns.countplot(x='addicted_label', data=df)
plt.title("Addiction Label Distribution")
plt.show()
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()
df.drop(['transaction_id','user_id'], axis=1, inplace=True)
X = df.drop('addicted_label', axis=1)
y = df['addicted_label']
X_train, X_test, y_train, y_test = train_test_split
X, y, test_size=0.2, random_state=42
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
