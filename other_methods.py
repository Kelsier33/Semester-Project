import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Load data
train_file_path = 'train_final.csv'
test_file_path = 'test_final.csv'

train_data = pd.read_csv(train_file_path)
test_data = pd.read_csv(test_file_path)

# Preprocess the data
common_columns = train_data.columns[1:-1].intersection(test_data.columns[1:])
X_train = train_data[common_columns].apply(pd.to_numeric, errors='coerce').fillna(0).values
y_train = train_data.iloc[:, -1].apply(pd.to_numeric, errors='coerce').fillna(0).values
X_test = test_data[common_columns].apply(pd.to_numeric, errors='coerce').fillna(0).values
test_ids = test_data.iloc[:, 0].apply(pd.to_numeric, errors='coerce').fillna(0).values

# Define the model
input_size = X_train.shape[1]

model = Sequential([
    Dense(50, activation='relu', input_shape=(input_size,)),  # Hidden Layer 1
    Dense(25, activation='relu'),                            # Hidden Layer 2
    Dense(1, activation='sigmoid')                           # Output Layer
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), 
              loss='binary_crossentropy', 
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

# Generate predictions
test_predictions = model.predict(X_test).flatten()

# Create submission file
submission_df = pd.DataFrame({
    "ID": test_ids,
    "Prediction": test_predictions
})
submission_file_path = 'submission_nn.csv'
submission_df.to_csv(submission_file_path, index=False)

print(f"Submission file saved to {submission_file_path}")
