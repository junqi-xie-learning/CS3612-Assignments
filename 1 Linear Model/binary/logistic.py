import numpy as np
from matplotlib import pyplot as plt
from util import *

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def logistic_loss(X, y, beta):
    return -np.sum(y * X.transpose().dot(beta) - np.log(1 + np.exp(X.transpose().dot(beta))))

def gradient(X, y, beta):
    return -X.dot(y - sigmoid(X.transpose().dot(beta)))

# Read the data from input file. Split it into 2 parts for training and testing.

data = np.loadtxt('Dataset_spine.csv', delimiter=',', skiprows=1, comments='"',
                  converters={6: lambda x: int(x != b'Abnormal'), 7: lambda x: 0})
train_data, test_data = split_data(data, 0.8)

# Retrieve the training data. Use gradient descent to train the model.

features_train, target_train = train_data[:, :6].transpose(), train_data[:, 6]
X_train = np.vstack((np.ones(features_train.shape[1]), features_train))
y_train = np.array(target_train)

beta = np.zeros(X_train.shape[0])
beta_trace = gd_const_ss(lambda beta: gradient(X_train, y_train, beta), beta, 2e-6, trace=True)
print(f'trained beta: {beta_trace[-1]}')

# Retrieve the testing data. Use the trained model to predict the value of attributes.

features_test, target_test = test_data[:, :6].transpose(), test_data[:, 6]
X_test = np.vstack((np.ones(features_test.shape[1]), features_test))
y_test = np.array(target_test)

def accuracy(X, y, beta):
    predicted = np.array([1 if x >= 0.5 else 0 for x in sigmoid(X.transpose().dot(beta))])
    return np.sum(predicted == y) / len(predicted)

acc = accuracy(X_test, y_test, beta_trace[-1])
print(f'prediction accuracy: {acc}')

# Plot changes in loss and accuracy.

train_loss = [logistic_loss(X_train, y_train, beta) for beta in beta_trace]
test_loss = [logistic_loss(X_test, y_test, beta) for beta in beta_trace]
train_acc = [accuracy(X_train, y_train, beta) for beta in beta_trace]
test_acc = [accuracy(X_test, y_test, beta) for beta in beta_trace]

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
