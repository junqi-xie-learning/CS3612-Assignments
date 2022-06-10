import numpy as np
# from matplotlib import pyplot as plt
# from sklearn.decomposition import PCA
from util import *

def seperate(X, y, type):
    positive, negative = [], []
    for i in range(len(y)):
        if y[i] == type:
            positive.append(X[:, i])
        else:
            negative.append(X[:, i])
    return positive, negative

# Read the data from input file. Split it into 2 parts for training and testing.

data = np.loadtxt('glass.csv', delimiter=',')
train_data, test_data = split_data(data, 0.9)
types = [1, 2, 3, 5, 6, 7]

# Retrieve the training data. Use linear discriminative analysis to analyze models
# for each type of glass.

features_train, type_train = train_data[:, 1:10].transpose(), train_data[:, 10]
X_train = np.array(features_train)
y_train = np.array(type_train)

models = []
for type in types:
    positive, negative = X_train[:, y_train == type], X_train[:, y_train != type]

    mu_pos, mu_neg = np.mean(positive, axis=1), np.mean(negative, axis=1)
    sigma_pos, sigma_neg = np.cov(positive), np.cov(negative)
    S_w = positive.shape[1] * sigma_pos + negative.shape[1] * sigma_neg

    beta = np.linalg.inv(S_w).dot(mu_pos - mu_neg)
    bias = (mu_pos + mu_neg) / 2
    models.append((beta, -bias.dot(beta)))

# Retrieve the testing data. Use the trained model to predict the classes.

features_test, type_test = test_data[:, 1:10].transpose(), test_data[:, 10]
X_test = np.array(features_test)
y_test = np.array(type_test)

for type in types:
    y_current = y_test.copy()
    for i in range(len(y_current)):
        y_current[i] = int(y_current[i] == type)

    beta, bias = models[types.index(type)]
    z = X_test.transpose().dot(beta) + bias
    predicted = z > 0

    accuracy = np.sum(predicted == y_current) / len(predicted)
    print(f'prediction accuracy for type {type}: {accuracy}')
