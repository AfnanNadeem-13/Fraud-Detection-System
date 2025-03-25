# Fraud-Detection-System

# Overview
This project implements a Fraud Detection System using machine learning techniques to identify fraudulent credit card transactions. The dataset used is the **Credit Card Fraud Dataset**, which contains transaction details along with labels indicating whether a transaction is fraudulent or not.

## Dataset
- **Source:** Kaggle (Credit Card Fraud Detection Dataset)
- **Features:** Transaction details (V1, V2, ..., V28), Time, Amount
- **Target:** `Class` (0 = Not Fraud, 1 = Fraud)
- **Size:** The dataset contains a large number of transactions, with an imbalance between fraud and non-fraud transactions.

## Installation & Setup
Clone the repository and install the required dependencies:

```sh
# Clone the repository
git clone https://github.com/AfnanNadeem-13/Fraud-Detection-System.git
cd Fraud-Detection-System

# Install dependencies
pip install -r requirements.txt
```

## Running the Project
Run the Python script to train and evaluate the fraud detection model:

```sh
python fraud_detection.py
```

## Steps Performed
1. **Data Preprocessing:**
   - Handled imbalanced data using SMOTE.
   - Normalized the features.
2. **Model Training:**
   - Trained a Random Forest classifier.
3. **Evaluation:**
   - Measured **Precision, Recall, and F1-score**.
4. **Testing Interface:**
   - Built a simple command-line interface for testing.

## Model Performance
- **Precision:** *xx.x%*
- **Recall:** *xx.x%*
- **F1-score:** *xx.x%*

## Usage
To test a transaction, use:

```sh
python test_transaction.py --amount 200 --time 3456 --features 0.1 0.2 ...
```

## Results & Visualizations
- Confusion Matrix
- ROC Curve
- Feature Importance Graph

## Repository Structure
```
Fraud-Detection-System/
│-- data/                     # Dataset files (excluded in GitHub)
│-- models/                   # Trained models
│-- fraud_detection.py        # Main script for training and evaluation
│-- test_transaction.py       # Script for testing transactions
│-- requirements.txt          # Required dependencies
│-- README.md                 # Project Documentation
```

## Submission Requirements Checklist
- [x] **GitHub Repository:** Code, dataset reference, and related files pushed.
- [x] **Visuals Submission:** Screenshots or a short video of results.
- [x] **Documentation:** `README.md` explaining project steps, execution, and observations.
- [x] **Submission Deadline:** March 27, 2025.

## References
- [Kaggle: Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- [Scikit-Learn Documentation](https://scikit-learn.org/)

## Author
Afnan Nadeem
