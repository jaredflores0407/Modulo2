import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import csv

def train_neural_network(X_train, y_train, X_test, y_test, hidden_size=5, learning_rate=0.000001, epochs=1100):
    # Initialize weights and biases
    input_size = X_train.shape[1]
    output_size = y_train.shape[1]
    weights_input_hidden = np.random.rand(input_size, hidden_size)
    bias_hidden = np.zeros((1, hidden_size))
    weights_hidden_output = np.random.rand(hidden_size, output_size)
    bias_output = np.zeros((1, output_size))

    # Training loop
    for epoch in range(epochs):
        # Forward propagation
        hidden_input = np.dot(X_train, weights_input_hidden) + bias_hidden
        hidden_output = 1 / (1 + np.exp(-hidden_input))  # Sigmoid activation for hidden layer
        final_input = np.dot(hidden_output, weights_hidden_output) + bias_output
        predicted_output = final_input  # Linear activation for output layer

        # Compute loss
        loss = 0.3 * np.mean((predicted_output - y_train) ** 2)

        # Backpropagation
        dloss_predicted = predicted_output - y_train
        dloss_hidden = np.dot(dloss_predicted, weights_hidden_output.T)
        dloss_hidden_output = dloss_predicted
        dloss_hidden_input = dloss_hidden * hidden_output * (1 - hidden_output)

        # Update weights and biases
        weights_hidden_output -= learning_rate * np.dot(hidden_output.T, dloss_hidden_output)
        bias_output -= learning_rate * np.sum(dloss_hidden_output, axis=0, keepdims=True)
        weights_input_hidden -= learning_rate * np.dot(X_train.T, dloss_hidden_input)
        bias_hidden -= learning_rate * np.sum(dloss_hidden_input, axis=0, keepdims=True)

        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss}')

    # Evaluate the model on the test set
    hidden_input_test = np.dot(X_test, weights_input_hidden) + bias_hidden
    hidden_output_test = 1 / (1 + np.exp(-hidden_input_test))
    final_input_test = np.dot(hidden_output_test, weights_hidden_output) + bias_output
    predicted_output_test = final_input_test
    test_loss = 0.3 * np.mean((predicted_output_test - y_test) ** 2)

    print(f'Test Loss: {test_loss}')

    # Define a predict function
    def predict(X_new):
        hidden_input = np.dot(X_new, weights_input_hidden) + bias_hidden
        hidden_output = 1 / (1 + np.exp(-hidden_input))
        final_input = np.dot(hidden_output, weights_hidden_output) + bias_output
        predicted_output = final_input
        return np.round(predicted_output).astype(int)

    return predict

train= pd.read_csv('train.csv')

X= train.drop('price_range', axis=1)
y= train['price_range'].values.reshape(-1,1)

# Generate sample data and split into train and test sets
np.random.seed(0)
#X = np.random.rand(100, 20)
#y = np.random.randint(0, 10, size=(100, 1)) 
# Normalize the features (X)
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=42)

# Example usage:
predictor = train_neural_network(X_train, y_train, X_test, y_test)