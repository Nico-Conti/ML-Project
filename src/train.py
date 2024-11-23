# src/train.py
import numpy as np
from nn_core import NeuralNetwork
from activation_function import ActivationReLU, ActivationSigmoid

from layer import DenseLayer
from sgd import SGD

def train_xor():
    # XOR dataset
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    y = np.array([
        [0],
        [1],
        [1],
        [0]
    ])

    # Create neural network
    nn = NeuralNetwork()
    nn.add(DenseLayer(2, 4, activation=ActivationReLU()))
    nn.add(DenseLayer(4, 1, activation=ActivationSigmoid()))

    # Create optimizer and loss function
    optimizer = SGD(learning_rate=0.1)
    loss_function = MeanSquaredError()

    # Train the network
    nn.train(X, y, epochs=10000, optimizer=optimizer, loss_function=loss_function)

if __name__ == "__main__":
    train_xor()