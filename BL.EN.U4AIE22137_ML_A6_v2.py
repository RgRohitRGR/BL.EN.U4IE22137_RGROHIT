import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier

# Function to predict output based on activation function
def predict(inp, weights, activationfunc):
    x = np.dot(inp, weights[1:]) + weights[0]
    if activationfunc == 1:
        return 1 if x >= 0 else 0  # Step function
    elif activationfunc == 2:
        return 1 if x >= 0 else -1  # Bipolar step function
    elif activationfunc == 3:
        return 1 / (1 + np.exp(-x))  # Sigmoid function
    elif activationfunc == 4:
        return max(0, x)  # ReLU function
    else:
        return -999999

# Function to calculate sum squared error
def sum_sq_err(a, b):
    return np.sum((a - b) ** 2)

# Function to classify using perceptron learning algorithm
def classifyq5(x, y, test, w):
    for epoch in range(1000):
        W = np.dot(x, w[1:]) + w[0]
        p = 1 / (1 + np.exp(-W))
        err = y - p
        adj = 0.05 * np.dot(x.T, err * p * (1 - p))
        w[1:] += adj
        w[0] += np.sum(adj)
    f = np.array(test)
    weightedsum = np.dot(f, w[1:]) + w[0]
    pred = 1 / (1 + np.exp(-weightedsum))
    if pred == 1:
        return "yes"
    else:
        return "no"

# Function to classify using pseudo-inverse
def classifyq6pinv(i, o, test):
    x = np.column_stack((np.ones(len(i)), i))
    p_inv = np.linalg.pinv(x)
    w = np.dot(p_inv, o)
    f = np.insert(test, 0, 1)
    weightedsum = np.dot(f, w)
    x = 1 / (1 + np.exp(-weightedsum))
    if x >= 0.5:
        return "yes"
    else:
        return "no"

# Function for perceptron learning
def perceptron_learning(inp, out, w, activationfunc, lr):
    err_val = []
    for epoch in range(1000):
        errors = []
        for i in range(len(inp)):
            pred = predict(inp[i], w, activationfunc)
            error = out[i] - pred
            errors += [error]
            w[1:] += lr * error * inp[i]
            w[0] += lr * error
        epoch_err = sum_sq_err(out, [predict(x, w, activationfunc) for x in inp])
        err_val += [epoch_err]
        if epoch_err <= 0.002:
            epoch = epoch + 1
            break
    return [epoch, err_val]

# Function for MLP classification
def mlp_class(X, Y):
    x = np.array(X)
    y = np.array(Y)
    mlp = MLPClassifier(hidden_layer_sizes=(2,), activation='logistic', solver='adam', max_iter=4000, random_state=42)
    mlp.fit(x, y)
    result = []
    for i in range(len(x)):
        result += [mlp.predict([x[i]])]
    return result

# Input data for logical gates
inp = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
ando = np.array([0, 0, 0, 1])
xoro = np.array([0, 1, 1, 0])
w = np.array([10, 0.2, -0.75])
gate = {'AND': ando, 'XOR': xoro}
act_func = ["step function", "bipolar step function", "sigmoid function", "relu function"]

# Plotting for convergence errors
for i in range(1, 5):
    ansand = perceptron_learning(inp, ando, w, i, 0.05)
    plt.plot(range(1, len(ansand[1]) + 1), ansand[1])
    plt.xlabel('epochs')
    plt.ylabel('sum square error')
    t = "AND error covergence using " + act_func[i - 1]
    plt.title(t)
    plt.show()
    ansxor = perceptron_learning(inp, xoro, w, i, 0.05)
    plt.plot(range(1, len(ansxor[1]) + 1), ansxor[1])
    plt.xlabel('epochs')
    plt.ylabel('sum square error')
    t = "XOR convergence error using " + act_func[i - 1]
    plt.title(t)
    plt.show()

# Plotting for XOR convergence errors again
for i in range(1, 5):
    ansxor = perceptron_learning(inp, xoro, w, i, 0.05)
    plt.plot(range(1, len(ansxor[1]) + 1), ansxor[1])
    plt.xlabel('epochs')
    plt.ylabel('sum square error')
    t = "XOR convergence error using " + act_func[i - 1]
    plt.title(t)
    plt.show()

# Plotting for learning rate vs. iterations to convergence
l_r = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
ita = []  # and
itx = []  # xor
for l in l_r:
    ita += [perceptron_learning(inp, ando, w, 1, l)[0]]
    itx += [perceptron_learning(inp, xoro, w, 1, l)[0]]
plt.plot(l_r, ita)
plt.xlabel('learning rate')
plt.ylabel('iteration to convergence')
plt.title('AND')
plt.show()
plt.plot(l_r, itx)
plt.xlabel('learning rate')
plt.ylabel('iteration to convergence')
plt.title('XOR')
plt.show()

# Data for classification
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
    [16, 2, 4, 198, 0]])
i = data[:, :-1]
o = data[:, -1]

# Classification using perceptron learning and pseudo-inverse
x = classifyq5(i, o, [18, 5, 4, 300], np.random.randn(5))
print("prediction~Is value high for [18,5,4,300]?", x)
x = classifyq6pinv(i, o, [18, 5, 4, 300])
print("pseudo inverse prediction~Is value high for [18,5,4,300]?", x)

# Classification using MLP
andmlp = mlp_class([[0, 0], [0, 1], [1, 0], [1, 1]], [0, 0, 0, 1])
i = [[0, 0], [0, 1], [1, 0], [1, 1]]
print("PREDICTED OUTPUT~")
print("AND")
for k in range(0, 4):
    print(i[k], " ", andmlp[k])
xormlp = mlp_class(i, [0, 1, 1, 0])
print("XOR")
for k in range(0, 4):
    print(i[k], " ", xormlp[k])
