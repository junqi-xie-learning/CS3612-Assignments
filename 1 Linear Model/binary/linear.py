import numpy as np
# from matplotlib import pyplot as plt
from util import *

def mse_loss(X, y, beta):
    diff = X.transpose().dot(beta) - y
    return diff.transpose().dot(diff) / X.shape[1]

def gradient(X, y, beta):
    diff = X.transpose().dot(beta) - y
    return X.dot(diff) / X.shape[1]

# Read the data from input file. Split it into 2 parts for training and testing.

data = np.loadtxt('index.txt', skiprows=1)
train_data, test_data = split_data(data, 0.9)
np.savetxt('data/train.txt', train_data, fmt="%.4f")
np.savetxt('data/test.txt', test_data, fmt="%.4f")

# Retrieve the training data. Use gradient descent to train the model.

age_train, fev_train = train_data[:, 0], train_data[:, 1]
X_train = np.vstack((np.ones(age_train.shape), age_train))
y_train = np.array(fev_train)

beta = np.zeros(X_train.shape[0])
beta = gd_const_ss(lambda beta: gradient(X_train, y_train, beta), beta, 0.002)
print(f'trained beta: {beta}')

# Retrieve the testing data. Use the trained model to predict the value of fev.

age_test, fev_test = test_data[:, 0], test_data[:, 1]
X_test = np.vstack((np.ones(age_test.shape), age_test))
y_test = np.array(fev_test)
err = mse_loss(X_test, y_test, beta) * X_test.shape[1]
print(f'prediction error: {err}')
