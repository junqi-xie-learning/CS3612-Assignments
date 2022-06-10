import numpy as np


def split_data(data, threshold):
    '''
    Split the data into training and testing sets.

    data: input dataset to be split
    threshold: the proportion of the dataset to include in the train split
    '''
    train_data, test_data = [], []
    for i in range(data.shape[0]):
        radio = np.random.random()
        if radio < threshold:
            train_data.append(data[i])
        else:
            test_data.append(data[i])
    return np.array(train_data), np.array(test_data)


def gd_const_ss(gradient, x0, stepsize, tol=1e-5, maxiter=100000, trace=False):
    '''
    Optimize the function using gradient decent algorithm with constant stepsize.

    gradient: function that takes an input x and returns the gradient of f at x
    x0: initial point in gradient descent
    stepsize: constant step size used in gradient descent
    tol: toleracne parameter in the stopping crieterion. Gradient descent stops 
         when the 2-norm of the gradient is smaller than tol
    maxiter: maximum number of iterations in gradient descent.
    '''
    x_traces = [np.array(x0)]
    x = np.array(x0)
    while np.linalg.norm(gradient(x)) >= tol and len(x_traces) <= maxiter:
        x -= stepsize * gradient(x)
        x_traces.append(np.array(x))
    return x_traces if trace else x
