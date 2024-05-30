import numpy as np


class Layer:
    def __init__(self):
        raise NotImplementedError

    def forward(self):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError


class Linear(Layer):
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.randn(output_size, input_size)
        self.biases = np.random.randn(output_size, 1)

    def forward(self, input):
        self.input = input
        return np.dot(self.weights, self.input) + self.biases

    def backward(self, output_gradient, learning_rate):
        weight_gradient = np.dot(output_gradient, np.transpose(self.input))
        self.weights -= learning_rate * weight_gradient
        self.biases -= learning_rate * output_gradient
        return np.dot(np.transpose(self.weights), output_gradient)


class Activation(Layer):
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, input):
        self.input = input
        return self.activation(self.input)

    def backward(self, output_gradient, learning_rate):
        return np.multiply(output_gradient, self.activation_prime(self.input))


class Sigmoid(Activation):
    def __init__(self):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        def sigmoid_prime(x):
            return (1 / (1 + np.exp(-x))) * (1 - (1 / (1 + np.exp(-x))))

        super().__init__(sigmoid, sigmoid_prime)


# TODO: Add ReLU(Activation)
