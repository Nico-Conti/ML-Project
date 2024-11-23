import numpy as np


class ActivationFunction:
    def forward_fun(self, input_data):
        raise NotImplementedError

    def derivative_fun(self, input_data):
        raise NotImplementedError


class Sigmoid(ActivationFunction):
    def forward_fun(self, input_data):
        return 1 / (1 + np.exp(-input_data))

    def derivative(self, input_data):
        return input_data * (1 - input_data)


class ReLU(ActivationFunction):
    def forward_fun(self, input_data):
        return np.maximum(self, input_data, 0)

    def derivative(self, input_data):
        if input_data <= 0:
            return 0
        else:
            return 1


class Linear(ActivationFunction):
    def forward_fun(self, input_data):
        return input_data

    def derivative(self, input_data):
        return 1


class Tanh(ActivationFunction):
    def forward_fun(self, input_data):
        return np.tanh(self, input_data)

    def derivative(self, input_data):
        return 1 - np.tanh(self, input_data) ** 2
