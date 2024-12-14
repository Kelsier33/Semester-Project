import numpy as np
import pandas as pd

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.layers = [input_size, hidden_size, hidden_size, output_size]
        # Initialize weights with input_size + 1 for bias
        self.weights = [
            np.random.randn(self.layers[i + 1], self.layers[i] + 1)
            for i in range(len(self.layers) - 1)
        ]

    def sigmoid(self, z):
        z = np.clip(z, -500, 500)  # Clip for numerical stability
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, z):
        s = self.sigmoid(z)
        return s * (1 - s)

    def forward(self, X):
        self.activations = [X]
        self.z_values = []
        for weight in self.weights:
            # Add bias dynamically during forward pass
            if X.shape[1] != weight.shape[1] - 1:  # Subtract 1 for bias
                continue
            X = np.c_[np.ones(X.shape[0]), X]  # Add bias
            z = np.dot(X, weight.T)
            self.z_values.append(z)
            X = self.sigmoid(z)
            self.activations.append(X)
        return self.activations[-1]

    def backward(self, X, y):
        m = X.shape[0]
        y = y.reshape(-1, 1)
        deltas = []

        # Output layer error
        delta = (self.activations[-1] - y) * self.sigmoid_derivative(self.z_values[-1])
        deltas.append(delta)

        # Hidden layers
        for l in range(len(self.weights) - 1, 0, -1):
            delta = np.dot(deltas[-1], self.weights[l][:, 1:]) * self.sigmoid_derivative(
                self.z_values[l - 1]
            )
            deltas.append(delta)

        deltas.reverse()

        # Gradients
        gradients = []
        for l in range(len(self.weights)):
            a = np.c_[np.ones(self.activations[l].shape[0]), self.activations[l]]
            grad = np.dot(deltas[l].T, a) / m
            gradients.append(grad)

        return gradients

    def update_weights(self, gradients, learning_rate):
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * gradients[i]

    def predict_proba(self, X):
        return self.forward(X)

# Load and preprocess data
train_file_path = 'train_final.csv'
test_file_path = 'test_final.csv'

train_data = pd.read_csv(train_file_path)
test_data = pd.read_csv(test_file_path)

# Preprocess data
X_train = train_data.iloc[:, 1:-1].apply(pd.to_numeric, errors='coerce').fillna(0).values
y_train = train_data.iloc[:, -1].apply(pd.to_numeric, errors='coerce').fillna(0).values
X_test = test_data.iloc[:, 1:].apply(pd.to_numeric, errors='coerce').fillna(0).values
test_ids = test_data.iloc[:, 0].apply(pd.to_numeric, errors='coerce').fillna(0).values

# Hyperparameters
hidden_size = 50
input_size = X_train.shape[1]
output_size = 1
gamma_0 = 1
d = 1000
epochs = 10

# Initialize and train the neural network
nn = NeuralNetwork(input_size, hidden_size, output_size)

t = 0
for epoch in range(epochs):
    permutation = np.random.permutation(len(X_train))
    X_train = X_train[permutation]
    y_train = y_train[permutation]

    for i in range(len(X_train)):
        x = X_train[i].reshape(1, -1)
        y = y_train[i].reshape(1, -1)
        learning_rate = gamma_0 / (1 + t * gamma_0 / d)
        t += 1

        nn.forward(x)
        gradients = nn.backward(x, y)
        nn.update_weights(gradients, learning_rate)

# Predict probabilities for test data
test_predictions = nn.predict_proba(X_test).flatten()

# Create submission file
submission_df = pd.DataFrame({
    "ID": test_ids,
    "Prediction": test_predictions
})
submission_file_path = 'submission_nn.csv'
submission_df.to_csv(submission_file_path, index=False)

print(f"Submission file saved to {submission_file_path}")




# What are some other methods of Machine Learning that have libraries that can quickly give me another submission file. 