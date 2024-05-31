import numpy as np


def mse(y_true, y_pred):
    # Generic formula for mean square error.
    return np.mean(np.power(y_true - y_pred, 2))


def mse_prime(y_true, y_pred):
    # The first derivative of the MSE formula.
    return 2 * (y_pred - y_true) / np.size(y_true)
