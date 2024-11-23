# src/layers/layer.py
import numpy as np
from activation_function import ActivationFunction

class Layer:
    """
    Base class for all layers in the neural network.
    """
    def forward(self, inputs):
        raise NotImplementedError("Subclasses should implement this method.")

    def backward(self, dvalues):
        raise NotImplementedError("Subclasses should implement this method.")


class DenseLayer(Layer):
    """
    Dense (fully connected) layer.
    """
    def __init__(self, n_inputs, n_neurons, activation: ActivationFunction = None):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        self.activation = activation

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

        if self.activation:
            self.output = self.activation.function(self.output)

        return self.output

    def backward(self, dvalues):
        if self.activation:
            dvalues = self.activation.derivative(self.output) * dvalues

        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        dinputs = np.dot(dvalues, self.weights.T)
        return dinputs