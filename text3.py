import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import shap

# Load dataset
dataset_path = "fashion-mnist.csv"  # Update if necessary
df = pd.read_csv(dataset_path)

# Extract features and labels
X = df.iloc[:, 1:].values  # Pixel values
y = df.iloc[:, 0].values   # Labels

# Normalize data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression model
model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
plt.figure(figsize=(10, 5))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

# Explainable AI (SHAP)
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test[:100])  # Explain first 100 test samples

# Summary plot for feature importance
shap.summary_plot(shap_values, X_test[:100], feature_names=[f"Pixel {i}" for i in range(X.shape[1])])
