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


def analyze_linear(svc, features):
    '''
    Analyze the linear model.
    '''
    print('\nLinear model analysis:')
    y = svc.support_vectors_
    weight = svc.dual_coef_[0]

    print(f'support vectors: {len(y)}')
    print(f'positive samples: {np.sum(weight > 0)}')
    print(f'negative samples: {np.sum(weight < 0)}')

    print('\nTop 30 support vectors:')
    indices = np.abs(weight).argsort()[::-1][:30]
    images = np.load('X_train_sampled.npy')

    for i in indices:
        support_vector = y[i]
        image_index = np.where((features == support_vector).all(1))[0][0]
        print(f'{image_index}: {weight[i]}')
        plt.imsave(f'support_vectors/{image_index}.png', images[image_index], cmap='gray')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # kernel: 'linear', 'rbf', 'poly'
    parser.add_argument('--kernel', type=str, default='linear', help='kernel specified for SVM')
    config = parser.parse_args()
    print(config)

    model, features = train(config.kernel)
    accuracy = test(model)
    print(f'prediction accuracy: {accuracy}')

    if config.kernel == 'linear':
        analyze_linear(model, features)
