import argparse
import numpy as np
from matplotlib import pyplot as plt
from sklearn.svm import SVC
from HoG import hog_features


def train(kernel):
    '''
    Load testing datasets from files. Use SVM to train the model.
    '''
    X_training = hog_features(np.load('X_train_sampled.npy'))
    y_training = np.load('y_train_sampled.npy')

    svc = SVC(kernel=kernel, C=1.5)  # C only applies to polynomial kernel
    svc.fit(X_training, y_training)

    return svc, X_training


def test(svc):
    '''
    Load testing datasets from files. Use the trained model to predict the classes.
    '''
    X_testing = hog_features(np.load('X_test_sampled.npy'))
    y_testing = np.load('y_test_sampled.npy')

    predicted = svc.predict(X_testing)
    return np.sum(predicted == y_testing) / len(predicted)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # kernel: 'linear', 'rbf', 'poly'
    parser.add_argument('--kernel', type=str, default='linear', help='kernel specified for SVM')
    config = parser.parse_args()
    print(config)

    model, features = train(config.kernel)
    accuracy = test(model)
    print(f'prediction accuracy: {accuracy}')
