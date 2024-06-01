def run(network, X, Y, error_func, error_prime, epochs, learning_rate):
    print()

    #Outermost for loop to go through the dataset "epoch" times.
    for epoch in range(epochs + 1):
        error = 0 # Initializing error to 0
        # Now iterating through every input and correct output in the dataset.
        for x, y in zip(X, Y):
            output = x
            # Taking each input and going forward through ech layer of the network (defined in main.py)
            for layer in network:
                output = layer.forward(output)
            # Accumulating onto the error
            error += error_func(y, output)
            # Finding gradient for a given input, then the below for loop iterates through each layer
            # backwards and makes the necessary changes to the weights and biases.
            gradient = error_prime(y, output)
            for layer in reversed(network):
                gradient = layer.backward(gradient, learning_rate)
        # Averaging out the error
        error /= len(X)
        if epoch % 10 == 0:
            print(f'Epoch: {epoch:3} | Error: {error}')
    print('\n---------------------------------------------\n')

    #Reset error and
    error = 0
    for x, y in zip(X, Y):
        output = x
        for layer in network:
            output = layer.forward(output)
        error += error_func(y, output)
    error /= len(X)
    print(f'Test Error: {error}\n')


def train(network, X, Y, error_func, error_prime, epochs, learning_rate):
    for epoch in range(epochs + 1):
        error = 0
        for x, y in zip(X, Y):
            output = x
            for layer in network:
                output = layer.forward(output)
            error += error_func(y, output)
            grad = error_prime(y, output)
            for layer in reversed(network):
                grad = layer.backward(grad, learning_rate)
        error /= len(X)
        if epoch % 10 == 0:
            print(f'Epoch: {epoch:3} | Error: {error}')


def test(network, X, Y, error_func):
    error = 0
    for x, y in zip(X, Y):
        output = x
        for layer in network:
            output = layer.forward(output)
        error += error_func(y, output)
    error /= len(X)
    print(f'Average Error: {error}')
