import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# 🔹 Step 1: Load Dataset
print("🔹 Loading dataset...")
df = pd.read_csv("data/creditcard.csv")  # Update path if needed

# 🔹 Step 2: Handle Class Imbalance
print("\n🔹 Handling class imbalance using SMOTE...")
X = df.drop(columns=["Class"])  # Features
y = df["Class"]  # Target variable

smote = SMOTE(sampling_strategy=0.1, random_state=42)  # 10% fraud samples
X_resampled, y_resampled = smote.fit_resample(X, y)

print(f"✅ New class distribution:\n{y_resampled.value_counts()}")

# 🔹 Step 3: Feature Scaling
print("\n🔹 Scaling features (Time, Amount)...")
scaler = StandardScaler()
X_resampled[["Time", "Amount"]] = scaler.fit_transform(X_resampled[["Time", "Amount"]])

# 🔹 Step 4: Train-Test Split
print("\n🔹 Splitting dataset into train & test sets...")
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

print(f"✅ Training set size: {X_train.shape[0]} samples")
print(f"✅ Testing set size: {X_test.shape[0]} samples")

# Train the model
print("\n Training the model")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("Model training completed!")

# Make predictions
print("\n Make predictions")
y_pred = model.predict(X_test)

# Evaluate the model
print("\n Model Evaluation:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# After training model,saving it
model_filename = "fraud_detection_model.pkl"
joblib.dump(model, model_filename)

print("f Model saved as {model_filename}")