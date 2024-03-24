#Q1.

import numpy as np
import matplotlib.pyplot as plt

# Initial weights
W = np.array([10, 0.2, -0.75])

# Learning rate
alpha = 0.05

# AND gate training data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 0, 0, 1])

# Step activation function
def step_function(x):
    return 1 if x >= 0 else 0

# Perceptron learning algorithm
def perceptron_learning(X, y, W, alpha, epochs):
    errors = []
    for epoch in range(epochs):
        total_error = 0
        for i in range(len(X)):# Forward pass
            prediction = step_function(np.dot(X[i], W[1:]) + W[0])  # Include bias term
            error = y[i] - prediction
            total_error += error ** 2
            W[1:] += alpha * error * X[i]  # Weight update
            W[0] += alpha * error  # Update bias separately
        errors.append(total_error)
        if total_error == 0:
            break
    return W, errors

# Train the perceptron
epochs = 100
W_final, error_values = perceptron_learning(X, y, W, alpha, epochs)

# Plot epochs against error values
plt.plot(range(1, len(error_values) + 1), error_values, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.title('Epochs vs Error')
plt.grid(True)
plt.show()

# Number of epochs needed for convergence
num_epochs_converged = len(error_values)
print("Number of epochs needed for weights to converge:", num_epochs_converged)



#Q3.
def perceptron_learning_with_iterations(X, y, W, alpha, max_epochs):
    n_features = X.shape[1]
    for epoch in range(max_epochs):
        converged = True
        for i in range(len(X)):
            # Forward pass
            prediction = step_function(np.dot(X[i], W[1:]) + W[0])  # Include bias term
            error = y[i] - prediction
            # Weight update
            W[1:] += alpha * error * X[i]
            W[0] += alpha * error  # Update bias separately
            if error != 0:
                converged = False
        if converged:
            return epoch + 1  # Return the number of iterations if converged
    return max_epochs  # Return max_epochs if not converged

# Learning rates
learning_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

# Number of iterations for each learning rate
num_iterations = []

# Perform perceptron learning for each learning rate
for alpha in learning_rates:
    W = np.array([10, 0.2, -0.75])  # Initial weights
    num_iter = perceptron_learning_with_iterations(X, y, W, alpha, max_epochs=1000)
    num_iterations.append(num_iter)

# Plotting
plt.plot(learning_rates, num_iterations, marker='o')
plt.xlabel('Learning Rate')
plt.ylabel('Number of Iterations')
plt.title('Number of Iterations vs Learning Rate')
plt.grid(True)
plt.show()



#Q4.
import numpy as np
import matplotlib.pyplot as plt

# Initial weights
W = np.array([10, 0.2, -0.75])

# Learning rate
alpha = 0.05

# AND gate training data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])

# Step activation function
def step_function(x):
    return 1 if x >= 0 else 0

# Perceptron learning algorithm
def perceptron_learning(X, y, W, alpha, epochs):
    errors = []
    for epoch in range(epochs):
        total_error = 0
        for i in range(len(X)):# Forward pass
            prediction = step_function(np.dot(X[i], W[1:]) + W[0])  # Include bias term
            error = y[i] - prediction
            total_error += error ** 2
            W[1:] += alpha * error * X[i]  # Weight update
            W[0] += alpha * error  # Update bias separately
        errors.append(total_error)
        if total_error == 0:
            break
    return W, errors

# Train the perceptron
epochs = 100
W_final, error_values = perceptron_learning(X, y, W, alpha, epochs)

# Plot epochs against error values
plt.plot(range(1, len(error_values) + 1), error_values, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.title('Epochs vs Error')
plt.grid(True)
plt.show()

# Number of epochs needed for convergence
num_epochs_converged = len(error_values)
print("Number of epochs needed for weights to converge:", num_epochs_converged)



#Q4.
def perceptron_learning_with_iterations(X, y, W, alpha, max_epochs):
    n_features = X.shape[1]
    for epoch in range(max_epochs):
        converged = True
        for i in range(len(X)):
            # Forward pass
            prediction = step_function(np.dot(X[i], W[1:]) + W[0])  # Include bias term
            error = y[i] - prediction
            # Weight update
            W[1:] += alpha * error * X[i]
            W[0] += alpha * error  # Update bias separately
            if error != 0:
                converged = False
        if converged:
            return epoch + 1  # Return the number of iterations if converged
    return max_epochs  # Return max_epochs if not converged

# Learning rates
learning_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

# Number of iterations for each learning rate
num_iterations = []

# Perform perceptron learning for each learning rate
for alpha in learning_rates:
    W = np.array([10, 0.2, -0.75])  # Initial weights
    num_iter = perceptron_learning_with_iterations(X, y, W, alpha, max_epochs=1000)
    num_iterations.append(num_iter)

# Plotting
plt.plot(learning_rates, num_iterations, marker='o')
plt.xlabel('Learning Rate')
plt.ylabel('Number of Iterations')
plt.title('Number of Iterations vs Learning Rate')
plt.grid(True)
plt.show()
