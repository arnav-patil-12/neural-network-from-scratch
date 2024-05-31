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

With the layers defined, we can now go ahead and code our error functions, which we use (for backpropogation), or to actually make our netowork "learn" from one training example to the next.

## [error_functions.py](error_functions.py)

For this simple neural network, I chose to include the most commonly-used error function, the mean squared error (or MSE). 

```
def mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))


def mse_prime(y_true, y_pred):
    return 2 * (y_pred - y_true) / np.size(y_true)
```

```mse_prime()``` is the first derivative of the MSE function. This is included because the netowrk relies on the derivative of the error function to calculate gradient (for gradient descent), which is further used to minimize the cost function.

With that, the error functions have been defined, and we can now make a separate file to define functions to actually manipulate the model.

## [run.py](run.py)

In this file, I will be defining three functions, ```run()```, ```train()```, and ```test()``` that I will import into [main.py](main.py) for simplicity. I will start with the ```train``` function. In this function, we want to loop over the training data set several times. We also want to define an error which we will initialize to zero. 

```
for epoch in range(epochs + 1):
        error = 0
```

We will then loop over each item in the training dataset. We want to run each item through the dataset, then calculate its error, and instruct the model to correct itself through backpropogation. This means that within the for loop defined above, we will have a nested for loop, which in turn contains anoter two for loops that iterate through each layer in the network. Starting with the iteration over each item in the dataset:

```
for x, y in zip(X, Y):
        output = x
```

Then, we can loop through each layer (forward), calculate the gradient descent, then go backwards through the model:

```
for layer in network:
    output = layer.forward(output)
error += error_func(y, output)
grad = error_prime(y, output)
for layer in reversed(network):
    grad = layer.backward(grad, learning_rate)
```

Lastly, we average out the error and print, marking the end of the function.

```
error /= len(X)
        if epoch % 10 == 0:
            print(f'Epoch: {epoch:3} | Error: {error}')
```

Now, we can assemble the ```test``` function, which contains essentially the same forward code, but no error backpropogation. For this function, iterating over each item in the dataset, then again through each layer of the network will suffice. 

```
def test(network, X, Y, error_func):
    error = 0
    for x, y in zip(X, Y):
        output = x
        for layer in network:
            output = layer.forward(output)
        error += error_func(y, output)
    error /= len(X)
    print(f'Average Error: {error}')
```

Lastly, we can consolidate this code into a single function called ```run```. 

```
def run(network, X, Y, error_func, error_prime, epochs, learning_rate):
    print()

    #for loop to train network
    for epoch in range(epochs + 1):
        error = 0
        for x, y in zip(X, Y):
            output = x
            for layer in network:
                output = layer.forward(output)
            error += error_func(y, output)
            gradient = error_prime(y, output)
            for layer in reversed(network):
                gradient = layer.backward(gradient, learning_rate)
        error /= len(X)
        if epoch % 10 == 0:
            print(f'Epoch: {epoch:3} | Error: {error}')
    print('\n---------------------------------------------\n')

    #for loop to test network
    error = 0
    for x, y in zip(X, Y):
        output = x
        for layer in network:
            output = layer.forward(output)
        error += error_func(y, output)
    error /= len(X)
    print(f'Test Error: {error}\n')
```
