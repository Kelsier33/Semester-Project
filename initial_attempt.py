# Re-import necessary libraries for a clean start
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score

# File paths for the training and test data
train_file_path = 'train_final.csv'
test_file_path = 'test_final.csv'

# Load the datasets
train_data = pd.read_csv(train_file_path)
test_data = pd.read_csv(test_file_path)

# Preprocessing function
def preprocess_data(data, is_training=True):
    data = data.replace('?', np.nan)
    for column in data.select_dtypes(include=['object']).columns:
        data[column].fillna(data[column].mode()[0], inplace=True)
    label_encoders = {}
    for column in data.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le
    if is_training:
        X = data.drop(columns=['income>50K'])
        y = data['income>50K']
        return X, y, label_encoders
    else:
        return data, label_encoders

# Preprocess the training data
X_train, y_train, label_encoders = preprocess_data(train_data)

# Split the data for training and validation
from sklearn.model_selection import train_test_split
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Initialize and train a RandomForestClassifier
model = RandomForestClassifier(random_state=42, n_estimators=100, class_weight='balanced')
model.fit(X_train_split, y_train_split)

# Validate performance on validation set
y_val_pred = model.predict_proba(X_val_split)[:, 1]
roc_auc = roc_auc_score(y_val_split, y_val_pred)

# Preprocess the test data
test_data_preprocessed, _ = preprocess_data(test_data, is_training=False)
test_data_preprocessed = test_data_preprocessed.drop(columns=['ID'])

# Predict on test data
test_predictions = model.predict_proba(test_data_preprocessed)[:, 1]

# Create submission file
submission_df = pd.DataFrame({
    'ID': test_data['ID'],
    'Prediction': test_predictions
})

# Save the submission file
submission_file_path = 'submission_improved.csv'
submission_df.to_csv(submission_file_path, index=False)

roc_auc, submission_file_path
