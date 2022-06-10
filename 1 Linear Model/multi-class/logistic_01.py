import numpy as np
# from matplotlib import pyplot as plt
# from sklearn.decomposition import PCA
from util import *

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def logistic_loss(X, y, beta):
    return -np.sum(y * X.transpose().dot(beta) - np.log(1 + np.exp(X.transpose().dot(beta))))

def gradient(X, y, beta):
    return -X.dot(y - sigmoid(X.transpose().dot(beta)))

# Read the data from input file. Split it into 2 parts for training and testing.

data = np.loadtxt('glass.csv', delimiter=',')
train_data, test_data = split_data(data, 0.9)
types = [1, 2, 3, 5, 6, 7]

# Retrieve the training data. Use gradient descent to train models for each type of glass.

features_train, type_train = train_data[:, 1:10].transpose(), train_data[:, 10]
X_train = np.vstack((np.ones(features_train.shape[1]), features_train))
y_train = np.array(type_train)

betas = []
for type in types:
    y_current = y_train.copy()
    for i in range(len(y_current)):
        y_current[i] = int(y_current[i] == type)

    beta = np.zeros(X_train.shape[0])
    beta = gd_const_ss(lambda beta: gradient(X_train, y_current, beta), beta, 2e-6)
    betas.append(beta)

# Retrieve the testing data. Use the trained model to predict the type based on the confidence score.

features_test, type_test = test_data[:, 1:10].transpose(), test_data[:, 10]
X_test = np.vstack((np.ones(features_test.shape[1]), features_test))
y_test = np.array(type_test)

confidence = np.array([sigmoid(X_test.transpose().dot(beta)) for beta in betas])
predicted = np.zeros(X_test.shape[1])
for i in range(X_test.shape[1]):
    predicted[i] = types[np.argmax(confidence[:, i])]

acc = np.sum(predicted == y_test) / X_test.shape[1]
print(f'prediction accuracy: {acc}')
