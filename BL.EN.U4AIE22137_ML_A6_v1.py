import numpy as np
import matplotlib.pyplot as plt

#1
class Perceptron:
    def __init__(self, input_size, learning_rate, initial_weights):
        self.weights = np.array(initial_weights)
        self.learning_rate = learning_rate
        self.errors = []

    def step_activation(self, x):
        return 1 if x >= 0 else 0

    def predict(self, inputs):
        weighted_sum = np.dot(inputs, self.weights[1:]) + self.weights[0]
        return self.step_activation(weighted_sum)

    def train(self, training_inputs, labels, max_epochs=100):
        for epoch in range(max_epochs):
            sum_square_error = 0
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                error = label - prediction
                sum_square_error += error ** 2
                self.weights[1:] += self.learning_rate * error * inputs
                self.weights[0] += self.learning_rate * error * 1
            self.errors.append(sum_square_error)
            if sum_square_error == 0:
                print(f"Converged in {epoch+1} epochs")
                break

initial_weights = [10, 0.2, -0.75]
learning_rate = 0.05

# AND gate truth table
training_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
labels = np.array([0, 0, 0, 1])

# Create and train the perceptron
perceptron = Perceptron(input_size=2, learning_rate=learning_rate, initial_weights=initial_weights)
perceptron.train(training_inputs, labels)

plt.plot(range(1, len(perceptron.errors) + 1), perceptron.errors)
plt.xlabel('Epochs')
plt.ylabel('Sum Square Error')
plt.title('Error vs Epochs')
plt.grid(True)
plt.show()


#3
def step_function(x):
    return 1 if x >= 0 else 0

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 0, 0, 1])

Wts = np.array([10, 0.2, -0.75])

# Maximum number of epochs
max_epochs = 1000

# Error convergence threshold
error_threshold = 0.002

# calculate the output of the perceptron
def predict(X, Wts):
    return step_function(np.dot(X, Wts[1:]) + Wts[0])

def sum_square_error(y_true, y_pred):
    return np.sum((y_true - y_pred) ** 2)

#perceptron learning with given weight
def perceptron_learning(X, y, Wts, alpha):
    for epoch in range(max_epochs):
        for i in range(len(X)):
            y_pred = predict(X[i], Wts)
            error = y[i] - y_pred
            Wts[1:] += alpha * error * X[i]
            Wts[0] += alpha * error
        epoch_error = sum_square_error(y, [predict(x, Wts) for x in X])
        if epoch_error <= error_threshold:
            return epoch + 1  # Return the number of iterations taken to converge
    return max_epochs  # If not converged, return maximum number of epochs

learning_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

iterations = []

for alpha in learning_rates:
    Wts = np.array([10, 0.2, -0.75])  
    iterations.append(perceptron_learning(X, y, Wts, alpha))

plt.plot(learning_rates, iterations, marker='o')
plt.xlabel('Learning Rate')
plt.ylabel('Iterations to Converge')
plt.title('Learning Rate vs Iterations to Converge')
plt.grid(True)
plt.show()

#5
import numpy as np

# Customer data
data = np.array([
    [20,6,2,386,1],   
    [16,3,6,289,1],  
    [27,6,2,393, 1],   
    [19,1,2,110, 0],  
    [24,4,2,280, 1],  
    [22,1,5,167, 0],  
    [15,4,2,271, 1],  
    [18,4,2,274, 1],  
    [21,1,4,148, 0],  
    [16,2,4,198, 0]   
])

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Initialize weights and bias
np.random.seed(0)  # for reproducibility
weights = np.random.randn(5)
learning_rate = 0.01
epochs = 1000

X = data[:, :-1]  # Features
y = data[:, -1]   # Labels

for epoch in range(epochs):
    # Forward pass
    weighted_sum = np.dot(X, weights[1:]) + weights[0]
    predictions = sigmoid(weighted_sum)
    
    # Backpropagation
    error = y - predictions
    adjustments = learning_rate * np.dot(X.T, error * predictions * (1 - predictions))
    
    weights[1:] += adjustments
    weights[0] += np.sum(adjustments)


def classify_transaction(customer):
    features = np.array(customer)
    weighted_sum = np.dot(features, weights[1:]) + weights[0]  # Include bias
    prediction = sigmoid(weighted_sum)
    if prediction >= 0.5:  
        return "High Value"
    else:
        return "Low Value"

test_customers = [
    [18, 5,4, 200],  #  high value 
    [10, 1,2, 100]   #  low value 
]
for customer in test_customers:
    prediction = classify_transaction(customer)
    print(f"Customer: {customer}, Predicted Probability: {prediction}")
for customer in test_customers:
    print(f"Customer: {customer}, Predicted Class: {classify_transaction(customer)}")

#6(not fully completed)
import numpy as np

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Pseudo-inverse method
def pseudo_inverse(X, y):
    X_with_bias = np.column_stack((np.ones(len(X)), X))  # Add bias term
    pseudo_inv = np.linalg.pinv(X_with_bias)
    weights = np.dot(pseudo_inv, y)
    return weights

# Classification function using weights obtained from pseudo-inverse
def classify_pseudo_inverse(customer, weights):
    features_with_bias = np.insert(customer, 0, 1)  # bias term
    weighted_sum = np.dot(features_with_bias, weights)
    return "High Value" if sigmoid(weighted_sum) >= 0.5 else "Low Value"

data = np.array([
    [20, 6, 2, 386, 1],   
    [16, 3, 6, 289, 1],   
    [27, 6, 2, 393, 1],   
    [19, 1, 2, 110, 0],   
    [24, 4, 2, 280, 1],   
    [22, 1, 5, 167, 0],   
    [15, 4, 2, 271, 1],   
    [18, 4, 2, 274, 1],   
    [21, 1, 4, 148, 0],   
    [16, 2, 4, 198, 0]    
])

X = data[:, :-1]  # Features
y = data[:, -1]   # Labels

weights_pseudo_inverse = pseudo_inverse(X, y)

test_customers = [
    [18, 5, 4, 200],  #  high value 
    [10, 1, 2, 100]   #  low value 
]

for customer in test_customers:
    pseudo_inverse_prediction = classify_pseudo_inverse(customer, weights_pseudo_inverse)
    print(f"Customer: {customer}, Pseudo-Inverse Prediction: {pseudo_inverse_prediction}")
