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
