# src/neural_network.py
from sgd import SGD


class NeuralNetwork:
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def backward(self, dvalues):
        for layer in reversed(self.layers):
            dvalues = layer.backward(dvalues)

    def train(self, X, y, epochs, optimizer, loss_function):
        for epoch in range(epochs):
            # Forward pass
            output = self.forward(X)

            # Compute loss
            loss = loss_function.function(y, output)

            # Backward pass
            dvalues = loss_function.derivative(y, output)
            self.backward(dvalues)

            # Update parameters
            for layer in self.layers:
                optimizer.update_params(layer)

            # Print epoch results
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}')