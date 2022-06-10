import numpy as np
from matplotlib import pyplot as plt
from util import *

def mse_loss(X, y, beta):
    diff = X.transpose().dot(beta) - y
    return diff.transpose().dot(diff) / X.shape[1]

def gradient(X, y, beta):
    diff = X.transpose().dot(beta) - y
    return X.dot(diff) / X.shape[1]

# Read the data from input file. Split it into 2 parts for training and testing.

data = np.loadtxt('forestfires.csv', delimiter=',', skiprows=1, comments=',,,')
train_data, test_data = data[:450], data[-67:]

# Retrieve the training and testing data.

attributes_train, area_train = train_data[:, :12].transpose(), train_data[:, 12]
X_train = np.vstack((np.ones(attributes_train.shape[1]), attributes_train))
y_train = np.log(np.array(area_train) + 1)

attributes_test, area_test = test_data[:, :12].transpose(), test_data[:, 12]
X_test = np.vstack((np.ones(attributes_test.shape[1]), attributes_test))
y_test = np.log(np.array(area_test) + 1)

# Use gradient descent to train the model. Use the trained model to predict the value of area.

beta = np.zeros(X_train.shape[0])
beta = gd_const_ss(lambda beta: gradient(X_train, y_train, beta), beta, 2e-6)
print(f'trained beta: {beta}')

err = mse_loss(X_test, y_test, beta) * X_test.shape[1]
print(f'prediction error: {err}')

# Visualize the weight of the model.

weight = np.sort(np.abs(beta[1:]))[::-1]
plt.plot(np.array(range(1, 13)), weight)
plt.savefig('graph/linear.pdf')
plt.close()
