import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import Perceptron
from scipy.stats import uniform

# AND gate training data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 0, 0, 1])

# Define parameter grid
param_dist = {'alpha': uniform(0.01, 0.1)}

# Create Perceptron model
perceptron = Perceptron(max_iter=100, tol=1e-3)

# Perform RandomizedSearchCV with 3-fold cross-validation
random_search = RandomizedSearchCV(perceptron, param_distributions=param_dist, n_iter=10, cv=3, random_state=42)
random_search.fit(X, y)

# Print results
print("Best learning rate found:", random_search.best_params_['alpha'])
print("Best score found:", random_search.best_score_)
