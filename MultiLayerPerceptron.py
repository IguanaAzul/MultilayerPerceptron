import numpy as np
from utils import function, not_implemented, error_function, d_error_function, d_funtion


class Layer:
    def __init__(self, n_inputs, n_neurons, activation="relu"):
        if activation == "step":
            print("Don`t use step function on MultiLayer perceptrons, "
                  "differentiable functions are needed for backpropagation.")
            print("Using relu instead.")
            activation = "relu"
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        self.activation = function[activation] if activation in function.keys() else not_implemented(activation)
        self.input_weights = np.random.rand(n_inputs, n_neurons) * 2 - 1
        self.biases = np.random.rand(n_neurons) * 2 - 1
        self.d_function = d_funtion[self.activation]

    def forward(self, inputs):
        out = np.dot(inputs, self.input_weights) + self.biases
        return self.activation(out), out


class Network:
    def __init__(self, n_inputs, lr=0.1):
        self.n_layers = 0
        self.n_outputs = 0
        self.n_inputs = n_inputs
        self.layers = list()
        self.lr = lr

    def add_layer(self, n_neurons, activation="step"):
        n_inputs = self.n_outputs if self.n_outputs != 0 else self.n_inputs
        self.n_outputs = n_neurons
        self.layers.append(Layer(n_inputs, n_neurons, activation))

    def predict(self, x):
        out = np.array(x)
        act = np.array(x)
        memory = dict()
        for idx, layer in enumerate(self.layers):
            memory["input_act_l" + str(idx)] = act
            memory["input_out_l" + str(idx)] = out
            act, out = layer.forward(out)
            memory["output_act_l" + str(idx)] = act
            memory["output_out_l" + str(idx)] = out
        return out, memory

    def print(self):
        for idx, layer in enumerate(self.layers):
            print("camada", idx)
            print("weights", layer.input_weights)
            print("biases", layer.biases)

    def predict_and_evaluate(self, x, y):
        x = np.array(x)
        y = np.array(y)
        guess, propagation_memory = self.predict(x)
        return guess, error_function(y, guess), propagation_memory

    def update(self, weight_gradients, bias_gradients):
        for idx, layer in enumerate(self.layers):
            layer.input_weights -= self.lr * weight_gradients[idx]
            layer.biases -= self.lr * bias_gradients[idx]

    def backpropagate(self, x, y):
        guess, error, propagation_memory = self.predict_and_evaluate(x, y)
        weight_gradients = dict()
        bias_gradients = dict()

        dError = d_error_function(y, guess)

        for idx, layer in reversed(list(enumerate(self.layers))):
            act = propagation_memory["output_act_l" + str(idx)]
            out = propagation_memory["output_out_l" + str(idx)]

            dOut = layer.d_function(dError, out)
            weight_gradients[idx] = np.dot(dOut, act.T) / layer.n_neurons
            bias_gradients[idx] = np.sum(dOut) / layer.n_neurons
            dError = np.dot(layer.input_weights, dOut)
        return weight_gradients, bias_gradients, guess, error, propagation_memory

    def train_and_evaluate(self, x_train, y_train, x_test, y_test, epochs):
        test_error = list()
        train_error = list()
        for epoch in range(epochs):
            for x, y in zip(x_train, y_train):
                weight_gradients, bias_gradients, guess, error, memory = self.backpropagate(x, y)
                self.update(weight_gradients, bias_gradients)
                train_error.append(error)
            for x, y in zip(x_test, y_test):
                t_guess, t_error, _ = self.predict_and_evaluate(x, y)
                test_error.append(t_error)
            print("Epoch ", epoch, " finished.")
            print("train_error: ", np.array(train_error).mean())
            print("test_error:", np.array(test_error).mean())
