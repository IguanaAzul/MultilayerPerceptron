import numpy as np


def step(x):
    x = np.array(x)
    x[x <= 0] = 0
    x[x > 0] = 1
    return x


def relu(x):
    x = np.array(x)
    x[x < 0] = 0
    return x


def sigmoid(x):
    x = np.array(x)
    return 1/(1+np.exp(-x))


def not_implemented(name):
    def ni(x):
        print("Activation function ", name, " not implemented")
        print("Using relu instead")
        return relu(x)
    return ni


function = {"step": step, "sigmoid": sigmoid, "relu": relu}


class Layer:
    def __init__(self, n_inputs, n_neurons, activation="step"):
        self.n_inputs = n_inputs
        self.activation = function[activation] if activation in function.keys() else not_implemented(activation)
        self.input_weights = np.random.rand(n_inputs, n_neurons) * 2 - 1
        self.biases = np.random.rand(n_neurons) * 2 - 1

    def forward(self, inputs):
        return self.activation(np.dot(inputs, self.input_weights) + self.biases)


class Network:
    def __init__(self, n_inputs):
        self.n_layers = 0
        self.n_outputs = 0
        self.n_inputs = n_inputs
        self.layers = list()

    def add_layer(self, n_neurons, activation="step"):
        n_inputs = self.n_outputs if self.n_outputs != 0 else self.n_inputs
        self.n_outputs = n_neurons
        self.layers.append(Layer(n_inputs, n_neurons, activation))

    def forward(self, inputs):
        inputs = np.array(inputs)
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def print(self):
        for idx, layer in enumerate(self.layers):
            print("camada", idx+1)
            print("weights", layer.input_weights)
            print("biases", layer.biases)

    def error(self, inputs, correct):
        inputs = np.array(inputs)
        correct = np.array(correct)
        guess = self.forward(inputs)
        return correct - guess
