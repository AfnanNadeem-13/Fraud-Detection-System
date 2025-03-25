import joblib
import numpy as np
import pandas as pd

# Load the trained model
model = joblib.load("fraud_detection_model.pkl")

# Load the dataset to get feature names
df = pd.read_csv("data/creditcard.csv")

# Drop unnecessary columns
features = df.drop(columns=["Class"]).columns

# Function to test the model
def test_transaction():
    print("\n🔹 Enter transaction details for fraud detection:")

    # Get user input for each feature
    input_data = []
    for feature in features:
        value = float(input(f"Enter value for {feature}: "))
        input_data.append(value)

    # Convert to NumPy array and reshape for prediction
    sample_transaction = np.array(input_data).reshape(1, -1)

    # Make prediction
    prediction = model.predict(sample_transaction)

    # Print result
    if prediction[0] == 1:
        print("\n⚠️ FRAUD DETECTED! ⚠️")
    else:
        print("\n✅ Legitimate Transaction")

# Run the function
if __name__ == "__main__":
    test_transaction()
