import numpy as np
import utilis.activation_function as af
# Create an instance of ActivationSigmoid
sigmoid_activation = af.ActivationSigmoid()

# Define some input data
input_data = np.array([0.5, -0.5, 1.0, -1.0])

# Compute the sigmoid activation
sigmoid_output = sigmoid_activation.sigmoid(input_data)
print("Sigmoid Output:", sigmoid_output)

# Compute the derivative of the sigmoid activation
sigmoid_derivative = sigmoid_activation.derivative(input_data)
print("Sigmoid Derivative:", sigmoid_derivative)