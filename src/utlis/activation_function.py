import numpy as np

class ActivationSigmoid:
    
    def __init__(self):
        self.function = None
        self.derivative = None

    def sigmoid(self, input):
        self.function = 1 / (1 + np.exp(-input))
        return self.function
    
    def derivative(self, input):
        if self.function is None:
            self.function = self.sigmoid(input)
        self.derivative = self.function * (1 - self.function)
        return self.derivative