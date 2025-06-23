import numpy as np
from .utils import dot_custom, sigmoid, sigmoid_derivative


# Single Neuron
class Neuron:
    def __init__(self, n_inputs):
        self.weights = np.random.randn(n_inputs)
        self.bias = np.random.randn()

    def forward(self, x):
        self.last_input = x
        self.last_z = dot_custom(self.weights, x) + self.bias
        self.output = sigmoid(self.last_z)
        return self.output

    def backward(self, dL_dy, learning_rate):
        dz = sigmoid_derivative(self.last_z) * dL_dy
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * dz * self.last_input[i]
        self.bias -= learning_rate * dz
    
# Fully Connected Layer
class Layer:
    def __init__(self, n_inputs, n_neurons):
        self.neurons = [Neuron(n_inputs) for _ in range(n_neurons)]

    def forward(self, x):
        self.last_input = x
        self.outputs = [neuron.forward(x) for neuron in self.neurons]
        return self.outputs

    def backward(self, dL_dy_list, learning_rate):
        for neuron, dL_dy in zip(self.neurons, dL_dy_list):
            neuron.backward(dL_dy, learning_rate)
    
# Multi-Layer Perceptron with Backprop
class MLP:
    def __init__(self, input_size, hidden_size, learning_rate=0.01):
        self.hidden = Layer(input_size, hidden_size)
        self.output = Neuron(hidden_size)
        self.lr = learning_rate

    def forward(self, x):
        self.hidden_output = self.hidden.forward(x)
        return self.output.forward(self.hidden_output)

    def compute_loss(self, y_true, y_pred):
        return (y_true - y_pred) ** 2

    def compute_loss_derivative(self, y_true, y_pred):
        return -2 * (y_true - y_pred)

    def train(self, X, y, epochs=100):
        for epoch in range(epochs):
            total_loss = 0
            for xi, yi in zip(X, y):
                y_pred = self.forward(xi)
                loss = self.compute_loss(yi, y_pred)
                total_loss += loss

                dL_dy = self.compute_loss_derivative(yi, y_pred)
                self.output.backward(dL_dy, self.lr)

                d_hidden = [
                    sigmoid_derivative(n.last_z) * dL_dy * self.output.weights[i]
                    for i, n in enumerate(self.hidden.neurons)
                ]
                self.hidden.backward(d_hidden, self.lr)

            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss:.4f}")