from layers import Linear, Sigmoid
from error_functions import mse, mse_prime
from run import run
import numpy as np


# XOR gate data
X = np.reshape([[0, 0], [0, 1], [1, 0], [1, 1]], (4, 2, 1))
Y = np.reshape([[0], [1], [1], [0]], (4, 1, 1))

# intialize the model
network = [
    Linear(2, 10),
    Sigmoid(),
    Linear(10, 10),
    Sigmoid(),
    Linear(10, 1),
    Sigmoid()
]

# hyperparameters
epochs = 100000
learning_rate = 0.1

#running the network
run(network, X, Y, mse, mse_prime, epochs, learning_rate)
