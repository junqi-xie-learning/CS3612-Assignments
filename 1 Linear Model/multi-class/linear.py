import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from util import *

def softmax(y):
    return np.exp(y - np.max(y)) / np.sum(np.exp(y - np.max(y)))

def cross_entropy_loss(X, y, W):
    return -np.sum(y * np.log(np.apply_along_axis(softmax, 1, X.transpose().dot(W))))

def gradient(X, y, W):
    return -X.dot(y - np.apply_along_axis(softmax, 1, X.transpose().dot(W)))

# Read the data from input file. Split it into 2 parts for training and testing.

data = np.loadtxt('glass.csv', delimiter=',')
train_data, test_data = split_data(data, 0.9)
types = [1, 2, 3, 5, 6, 7]

# Retrieve the training data. Convert the input label into distributions.
# Use gradient descent to train the model.

features_train, type_train = train_data[:, 1:10].transpose(), train_data[:, 10]
X_train = np.vstack((np.ones(features_train.shape[1]), features_train))
y_train = np.array(type_train)

y_dist = np.zeros((y_train.size, 6))
for i in range(y_train.size):
    index = types.index(y_train[i])
    y_dist[i][index] = 1

W = np.zeros((10, 6))
W = gd_const_ss(lambda W: gradient(X_train, y_dist, W), W, 2e-6)
print(f'trained W: {W}')

# Retrieve the testing data. Use the trained model to predict the type based on the confidence score.

features_test, type_test = test_data[:, 1:10].transpose(), test_data[:, 10]
X_test = np.vstack((np.ones(features_test.shape[1]), features_test))
y_test = np.array(type_test)

confidence = softmax(X_test.transpose().dot(W))
predicted = np.zeros(X_test.shape[1])
for i in range(X_test.shape[1]):
    predicted[i] = types[np.argmax(confidence[i, :])]
print(y_test)
print(predicted)

acc = np.sum(predicted == y_test) / X_test.shape[1]
print(f'prediction accuracy: {acc}')

# Plot the data with PCA.

pca = PCA(n_components=2)
W_pca = pca.fit_transform(W.transpose())
P = np.array(pca.components_)

plt.figure(figsize=(10, 8))
for i in range(6):
    plt.plot([0, W_pca[:, 0][i]], [0, W_pca[:, 1][i]])

X_pca = X_train.transpose() @ P.transpose()
plt.scatter(X_pca[:, 0], X_pca[:, 1], marker='o', c=y_train)
plt.savefig('graph/classification.pdf')
plt.show()
