import numpy as np
from matplotlib import pyplot as plt
from util import *

def rbf_kernel(X_i, X_j):
    K = np.zeros((X_i.shape[1], X_j.shape[1]))
    for i in range(X_i.shape[1]):
        for j in range(X_j.shape[1]):
            diff = (X_i[:, i] - X_j[:, j]) / 1000
            K[i, j] = diff.transpose().dot(diff)
    return np.exp(-K / 2).transpose()

def kernel_loss(K, y, c, lam):
    diff = K.transpose().dot(c) - y
    return diff.transpose().dot(diff) + lam * c.transpose().dot(K).dot(c)

def gradient(K, y, c, lam):
    diff = K.transpose().dot(c) - y
    return K.dot(diff) + lam * K.dot(c)

# Read the data from input file. Split it into 2 parts for training and testing.

data = np.loadtxt('forestfires.csv', delimiter=',', skiprows=1, comments=',,,')
train_data, test_data = data[:450], data[-67:]
lams = [0.001, 0.5, 10, 100]

# Retrieve the training and testing data.

attributes_train, area_train = train_data[:, :12].transpose(), train_data[:, 12]
X_train = attributes_train
y_train = np.log(np.array(area_train) + 1)

attributes_test, area_test = test_data[:, :12].transpose(), test_data[:, 12]
X_test = attributes_test
y_test = np.log(np.array(area_test) + 1)

# Train the model with different regulation terms. Use the trained model to predict the value of area.

models = []
for lam in lams:
    print(f'lambda = {lam}')

    c = np.zeros(X_train.shape[1])
    K = rbf_kernel(X_train, X_train)
    c = gd_const_ss(lambda c: gradient(K, y_train, c, lam), c, 2e-6)
    models.append(c)
    print(f'trained weights: {c}')

    predicted = c.dot(rbf_kernel(X_test, X_train))
    diff = predicted - y_test
    err = diff.transpose().dot(diff)
    print(f'prediction error: {err}')
    print()
