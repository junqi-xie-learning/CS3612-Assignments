import numpy as np
# from matplotlib import pyplot as plt
from util import *

def error(X, y, beta):
    return X.transpose().dot(beta) - y

def mse_loss(X, y, beta):
    diff = error(X, y, beta)
    return diff.transpose().dot(diff) / X.size

def gradient(X, y, beta):
    diff = error(X, y, beta)
    return X.dot(diff) / X.size

# Read the data from import file. Split it into 2 parts for training and testing.

data, labels = parse_txt("index.txt")
train_data, test_data = split_data(data, 0.9, 0.1)
output_data(train_data, "data/index_training.txt", labels, ' ')
output_data(test_data, "data/index_testing.txt", labels, ' ')

# Reteive the training data. Use gradient descent to train the model.

age_training, fev_training = train_data[:2]
X_training = np.array([[1.0] * len(age_training), age_training])
y_training = np.array(fev_training)

beta = np.array([0.0, 0.0]).transpose()
beta = gd_const_ss(lambda beta: gradient(X_training, y_training, beta), beta, 0.002)
print(f"trained beta: {beta}")

# Retrieve the testing data. Use the trained model to predict the value of fev.

age_testing, fev_testing = test_data[:2]
X_testing = np.array([[1.0] * len(age_testing), age_testing])
y_testing = np.array(fev_testing)
err = mse_loss(X_testing, y_testing, beta)
print(f"prediction error: {err}")
