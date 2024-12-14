import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

# File paths
train_file_path = 'train_final.csv'
test_file_path = 'test_final.csv'

# Load datasets
train_data = pd.read_csv(train_file_path)
test_data = pd.read_csv(test_file_path)

# Preprocess the data
common_columns = train_data.columns[1:-1].intersection(test_data.columns[1:])
X_train = train_data[common_columns].apply(pd.to_numeric, errors='coerce').fillna(0).values
y_train = train_data.iloc[:, -1].apply(pd.to_numeric, errors='coerce').fillna(0).values
X_test = test_data[common_columns].apply(pd.to_numeric, errors='coerce').fillna(0).values
test_ids = test_data.iloc[:, 0].apply(pd.to_numeric, errors='coerce').fillna(0).values

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define and train the logistic regression model
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Evaluate on training data (optional)
train_predictions = model.predict_proba(X_train)[:, 1]
train_auc = roc_auc_score(y_train, train_predictions)
print(f"Training AUC: {train_auc:.4f}")

# Predict probabilities for test data
test_predictions = model.predict_proba(X_test)[:, 1]

# Create submission file
submission_df = pd.DataFrame({
    "ID": test_ids,
    "Prediction": test_predictions
})
submission_file_path = 'submission_logistic_regression.csv'
submission_df.to_csv(submission_file_path, index=False)

print(f"Submission file saved to {submission_file_path}")
