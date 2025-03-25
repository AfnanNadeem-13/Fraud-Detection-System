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

| Metric      | Score  |
|------------|--------|
| **Precision** | 100%  |
| **Recall**    | 99%   |
| **F1-score**  | 99%   |

### Confusion Matrix:
[[56825 16] [ 47 5662]]

markdown
Copy
Edit

- **True Negatives (TN):** 56825 (Non-Fraud correctly classified)
- **False Positives (FP):** 16 (Non-Fraud wrongly classified as Fraud)
- **False Negatives (FN):** 47 (Fraud missed as Non-Fraud)
- **True Positives (TP):** 5662 (Fraud correctly detected)

### Key Observations:
- The model performs **exceptionally well**, detecting fraudulent transactions with **high accuracy**.
- Very **few false positives and false negatives**, indicating a **strong generalization**.
- The **recall for fraud cases is 99%**, meaning almost all fraud transactions are caught.

📌 **Next Steps:**
- Perform **cross-validation** to ensure model reliability.
- Generate an **ROC Curve & AUC Score** for deeper insights.


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
- [x] **Documentation:** `README.md` explaining project steps, execution.

## References
- [Kaggle: Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- [Scikit-Learn Documentation](https://scikit-learn.org/)

## Author
Afnan Nadeem
