
import numpy as np
import matplotlib.pyplot as plt
import scipy.io

def sigmoid(x):
    """
        Implements the sigmoid activation in numpy

        Arguments:
        x - numpy array of any shape

        Returns:
        output of sigmoid(x), same shape as x
    """

    return 1 / (1 + np.exp(-x))


def relu(x):
    """
        Implement the ReLU function.

        Arguments:
        x - numpy array of any shape

        Returns:
        output of ReLU(x), same shape as x
    """
    return np.maximum(0, x)


def tanh(x):
    """
        Implement the tanh function.

        Arguments:
        x - numpy array of any shape

        Returns:
        output of tanh(x), same shape as x
    """
    ex = np.exp(x)
    emx = np.exp(-x)
    return (ex - emx) / (ex + emx)


def softmax(Z):
    """
        Implement the softmax function.

        Arguments:
        Z - A matrix which the softmax function will be applied on each of it's column.

        Returns:
        output of softmax(Z).
    """


    eZ = np.exp(Z)
    return eZ / np.sum(eZ, axis=0, keepdims=True)


def softmax_gradient(softmax_result):
    """
        Compute the softmax gradient.

        Arguments:
            softmax_result - matrix

        Returns:
        output jacobian_matrix where:
            jacobian_matrix[i][j] = s[i] * (1-s[i]),  if i == j
            jacobian_matrix[i][j] = -s[i]*s[j]     ,  else
    """

    s = softmax_result.reshape(-1, 1)
    return np.diagflat(s) - np.dot(s, s.T)


def compute_cost(Y_hat, Y):
    """
    Implement the cost function.

    Arguments:
    Y_hat -- probability metrics corresponding to your label predictions, shape (mini_batch_size, classes)
    Y -- Ground-truth labels, shape (mini_batch_size, classes)

    Returns:
    cost -- cross-entropy cost
    """

    m = Y.shape[1]


    cost = (1. / m) * np.sum(np.multiply(-np.log(Y_hat + 0.000000001),Y) + np.multiply(-np.log(1 - Y_hat + 0.000000001), 1 - Y))

    cost = np.squeeze(cost)  # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).

    return cost


def load_dataset(file_name):
    data = scipy.io.loadmat(file_name)
    train_X = data['Yt']
    train_Y = data['Ct']
    test_X = data['Yv']
    test_Y = data['Cv']

    return train_X, train_Y, test_X, test_Y

