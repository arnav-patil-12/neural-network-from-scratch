# Python File Documentation

As promised in [README.md](README.md), this document is meant to provide an overview of my code my understanding of it as I followed through the video playlist.

## [layers.py](layers.py)

This was the first file I created, because it's the first one the video creates as well. I began by creating an abstract class ```Layer``` which I will use as the basis for all my layers. The ```NotImplementedError``` is raised to signify that ```Layer``` is an abstract class and should not be initalized as it is.

```
class Layer:
    def __init__(self):
        raise NotImplementedError

    def forward(self):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError
```

### The Linear (or Dense) Layer

To define the dense layer, we must give it an input and output size (as in how many "neurons" feed into it, and how many neurons it feeds into). Each layer also has its quantities called weights and biases (discussed in README), which are quantities that should be initalized as well.

```
class Linear(Layer):
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.randn(output_size, input_size)
        self.biases = np.random.randn(output_size, 1)
```

We then also define the forward and backward calculations of the linear layer. These are just the computations defined in the README (and videos) translated into Python-speak.

```
def forward(self, input):
        self.input = input
        return np.dot(self.weights, self.input) + self.biases

    def backward(self, output_gradient, learning_rate):
        weight_gradient = np.dot(output_gradient, np.transpose(self.input))
        self.weights -= learning_rate * weight_gradient
        self.biases -= learning_rate * output_gradient
        return np.dot(np.transpose(self.weights), output_gradient)
```

### Activation Layers

In the next section of this Python file, we will define the activation layer and some commonly-used functions. These layers are transformations that allow the network to deal in non-linear relationships as well. The activation functions I chose to include are tanh, the sigmoid, and ReLU.

```
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


class ReLU(Activation):
    def __init__(self):
        def relu(x):
            return np.maximum(0, x)

        def relu_prime(x):
            return np.where(x > 0, x, 0)
```
