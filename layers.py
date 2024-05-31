import numpy as np


class Layer:
    # Since this is an abstract class, we cannot initialize a Layer object on its own.
    # To mitigate this, we raise a NotImplementedError when initialized.
    def __init__(self):
        raise NotImplementedError

    def forward(self):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError


class Linear(Layer):
    def __init__(self, input_size, output_size):
        # Size attributed based the data that will be passed into this layer,
        # and what layer it will pass data into
        self.input_size = input_size
        self.output_size = output_size
        # Weights and biases assigned randomly upon initialization.
        self.weights = np.random.randn(output_size, input_size)
        self.biases = np.random.randn(output_size, 1)

    def forward(self, input):
        self.input = input
        # Formula for this neuron's activation
        return np.dot(self.weights, self.input) + self.biases

    def backward(self, output_gradient, learning_rate):
        # All the fancy pants math discussed in the video that essentially changes the weights and biases
        # of the layer to "learn" as it progresses through the dataset.
        weight_gradient = np.dot(output_gradient, np.transpose(self.input))
        self.weights -= learning_rate * weight_gradient
        self.biases -= learning_rate * output_gradient
        return np.dot(np.transpose(self.weights), output_gradient)


class Activation(Layer):
    def __init__(self, activation, activation_prime):
        # Layers that essentially squish the output of the previous Linear layer into a
        # value between 0 and 1, which will be fed into the next Linear layer.
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, input):
        self.input = input
        return self.activation(self.input)

    def backward(self, output_gradient, learning_rate):
        return np.multiply(output_gradient, self.activation_prime(self.input))


class Sigmoid(Activation):
    # Another name for the logistic function??????
    def __init__(self):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        def sigmoid_prime(x):
            return (1 / (1 + np.exp(-x))) * (1 - (1 / (1 + np.exp(-x))))

        super().__init__(sigmoid, sigmoid_prime)


class ReLU(Activation):
    # Another, more popular activation function that is essentially either on or off.
    def __init__(self):
        def relu(x):
            return np.maximum(0, x)

        def relu_prime(x):
            return np.where(x > 0, x, 0)
