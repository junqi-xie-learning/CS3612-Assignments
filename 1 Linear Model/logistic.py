import numpy as np
from matplotlib import pyplot as plt
from util import *

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def logistic_loss(X, y, beta):
    return -np.sum(y * X.transpose().dot(beta) + np.log(1 + np.exp(X.transpose().dot(beta))))

def gradient(X, y, beta):
    return X.dot(sigmoid(X.transpose().dot(beta)) - y)

def predict(X, beta):
    original = sigmoid(X.transpose().dot(beta))
    for i in range(len(original)):
        if original[i] > 0.5:
            original[i] = 1
        else:
            original[i] = 0
    return original

def accuracy(X, y, beta):
    return 1 - np.sum(np.abs(predict(X, beta) - y)) / X.size

# Read the data from import file. Split it into 2 parts for training and testing.

data, labels = parse_csv("Dataset_spine.csv")
train_data, test_data = split_data(data, 0.8, 0.2)
output_data(train_data, "data/spine_training.csv", labels, ',')
output_data(test_data, "data/spine_testing.csv", labels, ',')

# Reteive the training data. Use gradient descent to train the model.

cols_training, att_training = train_data[:12], train_data[12]
X_training = np.array([[1.0] * len(cols_training[0])] + cols_training)
y_training = np.array(att_training)

beta = np.array([0.0] * 13)
beta_trace = gd_const_ss(lambda beta: gradient(X_training, y_training, beta), beta, 2e-6, trace=True)
print(f"trained beta: {beta_trace[-1]}")

# Retrieve the testing data. Use the trained model to predict the attribute.

cols_testing, att_testing = test_data[:12], test_data[12]
X_testing = np.array([[1.0] * len(cols_testing[0])] + cols_testing)
y_testing = np.array(att_testing)
err = 1 - accuracy(X_testing, y_testing, beta_trace[-1])
print(f"prediction error: {err}")

# Plot changes in loss and accuracy.

train_loss = list(map(lambda beta: logistic_loss(X_training, y_training, beta), beta_trace))
test_loss = list(map(lambda beta: logistic_loss(X_testing, y_testing, beta), beta_trace))
train_acc = list(map(lambda beta: accuracy(X_training, y_training, beta), beta_trace))
test_acc = list(map(lambda beta: accuracy(X_testing, y_testing, beta), beta_trace))

plt.plot(train_loss, label='train loss')
plt.savefig('graph/train_loss.pdf')
plt.close()

plt.plot(test_loss, label='test loss')
plt.savefig('graph/test_loss.pdf')
plt.close()

plt.plot(train_acc, label='train accuracy')
plt.savefig('graph/train_acc.pdf')
plt.close()

plt.plot(test_acc, label='test accuracy')
plt.savefig('graph/test_acc.pdf')
plt.close()
