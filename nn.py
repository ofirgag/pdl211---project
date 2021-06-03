import numpy as np
import matplotlib.pyplot as plt

from nn_utils import *

def update_parameters(parameters, learning_rate, grads):
    """

    :param parameters: W and b for each layer l in [1, L-1]
    :param learning_rate: Step size
    :param grads: Gradients of W_l and b_l for each layer l in [1, L-1]
    :return: Updated parameters
    """

    L = len(parameters) // 2  # number of layers in the neural network

    for l in range(L):
        parameters['W' + str(l + 1)] -= learning_rate * grads['dW' + str(l + 1)]
        parameters['b' + str(l + 1)] -= learning_rate * grads['db' + str(l + 1)]

    return parameters


def SGD(X, Y, layers_dims, initialization_param=2, epochs=5, learning_rate=0.01, mini_batch_size=16):
    """
        Approximate W that minimizes f(W, (X, Y))

        Arguments:
        X - input data. dim = [n, m]
        Y - Ground truth labels
        layers_dims - Network architecture
        initialization_param - A parameter used for Xavier and He initialization (1 for Xavier, 2 for He)
        epochs - number of iteration over all X
        learning_rate - step size
        mini_batch_size - number of examples in each of the mini-batches

        Returns:
        parameters - A dictionary contains the weights and the biases learned in SGD.
    """
    costs = []
    m = X.shape[1]
    # Initialize parameters
    parameters = {}
    L = len(layers_dims)

    for l in range(1, L):
        # He initialization
        parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1]) * np.sqrt(initialization_param / layers_dims[l - 1])
        parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))

    for epoch in range(epochs):

        mini_batches = build_mini_batches(X, Y, m, mini_batch_size)
        cost = 0
        for mb in mini_batches:
            # Extract mini_batches
            mini_batch_X, mini_batch_y = mb

            # Forward propagation
            Y_hat, caches = forward_propagation(mini_batch_X, parameters)

            # Compute_cost
            cost += compute_cost(Y_hat, mini_batch_y)

            # Backward propagation
            grads = backward_propagation(Y_hat, mini_batch_y, caches)

            # Update parameters
            parameters = update_parameters(parameters, learning_rate, grads)
        costs.append(cost)
        if(len(costs) % 50 == 0):
            print(cost)
    return parameters

def build_mini_batches(X, Y, m, mini_batch_size):
    """
        Define the random minibatches. We increment the seed to reshuffle differently the dataset after each epoch

    :param X: Input examples
    :param Y: True labels
    :param m: Number of examples in the input.
    :param mini_batch_size: The size of each mini-batch.
    :return: A list of the mini-batches
    """

    mini_batch_indices = [i for i in range(m)]
    np.random.shuffle(mini_batch_indices)
    shuffled_X = X[:, mini_batch_indices]
    shuffled_y = Y[:, mini_batch_indices]
    mini_batches_X = [(shuffled_X[:, i: i + mini_batch_size]) for i in range(0, m, mini_batch_size)]
    mini_batches_y = [(shuffled_y[:, i: i + mini_batch_size]) for i in range(0, m, mini_batch_size)]
    if (m % mini_batch_size) != 0:
        mini_batches_X.append(shuffled_X[:, m - (m % mini_batch_size): m])
        mini_batches_y.append(shuffled_y[:, m - (m % mini_batch_size): m])
    mini_batches = [(mini_batches_X[i], mini_batches_y[i]) for i in range(len(mini_batches_y))]
    return mini_batches


def forward_propagation(x, parameters, activation=relu):
    """
    :param x: Input example
    :param parameters: Dictionary of weights W and biases b
    :param activation: activation function
    :return: argmax(softmax(Wx + b))
    """
    caches = []
    L = len(parameters) // 2
    out_prev = x
    for l in range(1, L):
        Wl = parameters['W' + str(l)]
        bl = parameters['b' + str(l)]
        Z = np.dot(Wl, out_prev) + bl
        caches.append((Z, out_prev, Wl, bl))
        out_prev = activation(Z)

    WL = parameters['W' + str(L)]
    bL = parameters['b' + str(L)]
    Z = np.dot(WL, out_prev) + bL
    out = softmax(Z)
    caches.append((Z, out_prev, WL, bL))

    return out, caches


def compute_gradients(cache, Y_hat=None, Y=None, activation='relu', dout=None):
    """
    :param cache: Saved computations from the forward propagation.
    :param Y_hat: The output from the forward propagation (used for softmax).
    :param Y: True labels.
    :param activation: The activation function used for the required layer.
    :param dout: The gradient of the loss function with respect to the output.
    :return: The gradients of the loss with respect to the output from previous layer, to W and to b.
    """

    Z, out_prev, W, b = cache
    m = out_prev.shape[1]
    dZ = 0
    if activation == 'relu':
        dZ = np.array(dout, copy=True)  # just converting dz to a correct object.
        # When z <= 0, you should set dz to 0 as well.
        dZ[Z <= 0] = 0

    elif activation == 'tanh':
        dZ = np.multiply(dout, 1 - (tanh(Z) ** 2))
    elif activation == 'softmax':
        dZ = Y_hat - Y

    dout_prev = np.dot(W.T, dZ)
    dW = 1. / m * np.dot(dZ, out_prev.T)
    db = 1. / m * np.sum(dZ, axis=1, keepdims=True)

    return dout_prev, dW, db


def backward_propagation(Y_hat, Y, caches):
    """
    :param Y_hat: The result of the forward_propagation
    :param Y: Ground truth labels
    :param caches: Saved computations from the forward_propagation
    :return: gradients of Wl, bl and out_l for each layer l
    """

    grads = {}
    L = len(caches)

    dout_prev, dWL, dbL = compute_gradients(caches[L - 1], Y_hat, Y, 'softmax')
    grads['dout' + str(L - 1)] = dout_prev
    grads['dW' + str(L)] = dWL
    grads['db' + str(L)] = dbL

    for l in range(L - 1, 0, -1):
        dout_prev, dWl, dbl = compute_gradients(caches[l - 1], dout=grads['dout' + str(l)])
        grads['dout' + str(l - 1)] = dout_prev
        grads['dW' + str(l)] = dWl
        grads['db' + str(l)] = dbl

    return grads


if __name__ == '__main__':
    train_X, train_Y, test_X, test_Y = load_dataset('PeaksData.mat')
    layers_dims = [2, 20, 15, 10, 7, 6]
    parameters = SGD(train_X, train_Y, layers_dims, learning_rate=0.001, mini_batch_size=256, epochs=200)
