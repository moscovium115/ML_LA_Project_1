import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix
from scipy.special import dtype


def load_data(file_path):
    """Loads the data matrix X and target vector y from a CSV file."""
    # Read CSV file
    csv_data = pd.read_csv(file_path)
    csv_data = csv_data.astype(np.float256)

    # Separate features (X) and target (y)
    X = csv_data.iloc[:, :-1].values
    y = csv_data.iloc[:, -1].values

    # Convert y to ±1 (assuming original y is in {0, 1})
    y = np.where(y == 0, -1, 1)

    return X, y


# Load the dataset
X, y = load_data('data.csv')

# Display data details
print(f"Loaded dataset with {X.shape[0]} samples and {X.shape[1]} features.")
print(f"Number of malicious data points: {np.sum(y == 1)}")
print(f"Number of non-malicious data points: {np.sum(y == -1)}")

np.random.seed(412)  # For reproducibility


def split_data(X, y, r=0.5):
    """Splits the data into training and test sets based on r, the test_size ratio."""
    indices = np.random.permutation(len(X))
    split_index = int(len(X) * (1 - r))

    X_train = X[indices[:split_index]]
    y_train = y[indices[:split_index]]
    X_test = X[indices[split_index:]]
    y_test = y[indices[split_index:]]

    return X_train, X_test, y_train, y_test


# Split the dataset (50/50 split)
X_train, X_test, y_train, y_test = split_data(X, y, r=0.50)

def classify(X, y, w):
    """Returns the number of correctly classified points using the weight vector w."""
    print(np.dot(X, w))
    predictions = np.sign(np.dot(X, w))
    #list of 0,1 and 1 if the values are the same, so by summing this up we get number of correctly identified datapoints
    correct = np.sum(predictions == y)
    accuracy = correct / len(y)
    return correct, accuracy

# Example: Try random weight vector
w_random = np.random.randn(X_train.shape[1])
correct, accuracy = classify(X_test, y_test, w_random)
print(f"Random classification accuracy: {accuracy * 100:.2f}%")


def sigmoid(z):
    """Sigmoid function."""
    return 1 / (1 + np.exp(-z))

#X is sparse and has much less columns than rows,
X_sparse=csc_matrix(X_train)

def logistic_regression_cost_grad(X, y, w, reg_lambda):
    """Calculates the cost and gradient for logistic regression."""
    m = len(y)
    z = X_sparse.dot(w)
    # z= np.dot(X,w)
    h = sigmoid(z)

    # print("debug")
    # print(np.dot(X,w))


    # Cost function with regularization
    cost = (-1 / m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h)) + (reg_lambda / (2 * m)) * np.dot(w, w)
    print(h[0])
    # Gradient with regularization

    grad = (1 / m) *  X_sparse.transpose().dot(h-y) + (reg_lambda / m) * w
    # grad = (1 / m) * np.dot(X.T, (h - y)) + (reg_lambda / m) * w


    return cost, grad


def logistic_regression(X, y, alpha, reg_lambda, num_steps):
    """Performs gradient descent to find the optimal weight vector for logistic regression."""
    # Initialize weight vector w
    w = np.zeros(X.shape[1])

    for step in range(num_steps):
        cost, grad = logistic_regression_cost_grad(X, y, w, reg_lambda)
        w -= alpha * grad  # Gradient descent update

        # Optional: print progress
        if step % 100 == 0:
            print(f"Step {step}, Cost: {cost}")

    return w


# Train logistic regression with gradient descent
w_trained = logistic_regression(X_train, y_train, alpha=0.13, reg_lambda=0.1, num_steps=10000)

# Evaluate performance on the test set
correct, accuracy = classify(X_test, y_test, w_trained)
print(f"Test set classification accuracy: {accuracy * 100:.2f}%")

#kan niet boven 70% krijgen met 50/50 split van datapunten, hoger en ik krijg error

from scipy.linalg import svd

def detect_remove_outliers(X, threshold=1e-2):
    """Detects and removes outliers based on singular values."""
    U, S, Vt = svd(X, full_matrices=False)
    mask = S > threshold
    X_cleaned = X[:, mask]
    print(f"Removed {X.shape[1] - X_cleaned.shape[1]} outliers")
    return X_cleaned

# Apply SVD to remove outliers


X_cleaned = detect_remove_outliers(X)
