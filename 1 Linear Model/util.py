import numpy as np


def parse_txt(filename):
    """
    Parse the text file and return a list of input data.
    """
    data = []
    labels = []
    with open(filename, 'r') as f:
        for label in f.readline().strip().split():
            labels.append(label)
            data.append([])
        for line in f:
            line = line.strip().split()
            for i, num in enumerate(line):
                data[i].append(float(num))
    return data, labels


def output_data(data, filename, labels, spliter):
    """
    Output the data into a file.
    """
    with open(filename, 'w') as f:
        f.write(spliter.join(labels) + '\n')
        length = len(data[0])
        for i in range(length):
            f.write(spliter.join(str(line[i]) for line in data) + '\n')


def split_data(data, train, test):
    """
    Split the data into training and testing sets.
    """
    train_data = []
    test_data = []
    for _ in data:
        train_data.append([])
        test_data.append([])

    for i in range(len(data[0])):
        radio = np.random.random()
        if radio < train:
            for j, line in enumerate(data):
                train_data[j].append(line[i])
        elif radio > 1 - test:
            for j, line in enumerate(data):
                test_data[j].append(line[i])
    return train_data, test_data


def gd_const_ss(gradient, x0, stepsize, tol=1e-5, maxiter=100000, trace=False):
    """
    Optimize the function using gradient decent algorithm with constant stepsize.

    gradient: function that takes an input x and returns the gradient of f at x
    x0: initial point in gradient descent
    stepsize: constant step size used in gradient descent
    tol: toleracne parameter in the stopping crieterion. Gradient descent stops 
         when the 2-norm of the gradient is smaller than tol
    maxiter: maximum number of iterations in gradient descent.
    """
    x_traces = [np.array(x0)]
    x = np.array(x0)
    while np.linalg.norm(gradient(x)) >= tol and len(x_traces) <= maxiter:
        x -= stepsize * gradient(x)
        x_traces.append(np.array(x))
    return x_traces if trace else x
