import numpy as np
from utils import function, not_implemented, sigmoid
from sklearn.metrics import accuracy_score


class Perceptron:
    def __init__(self, n_inputs, n_neurons, lr=0.1, activation="step"):
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        self.activation = function[activation] if activation in function.keys() else not_implemented(activation)
        self.weights = np.random.rand(n_inputs, n_neurons) * 2 - 1
        self.biases = np.random.rand(n_neurons) * 2 - 1
        self.lr = lr

    def predict(self, x):
        out = self.activation(np.dot(x, self.weights) + self.biases)
        if self.activation == sigmoid:
            argmax = np.argmax(out)
            out = np.zeros(len(out))
            out[argmax] = 1
        return out

    def predict_and_evaluate(self, x, y):
        predicted = self.predict(x)
        return predicted, y - predicted

    def update(self, x, y):
        predicted, error = self.predict_and_evaluate(x, y)
        for idx, e in enumerate(error):
            self.weights[:, idx] += self.lr * e * x
            self.biases[idx] += self.lr * e * x[idx]
        return predicted, error

    def train_and_evaluate(self, x_train, y_train, x_test, y_test, epochs):
        for epoch in range(epochs):
            train_predictions = list()
            test_predictions = list()
            for x, y in zip(x_train, y_train):
                predicted, _ = self.update(x, y)
                train_predictions.append(predicted)
            for x, y in zip(x_test, y_test):
                predicted, _ = self.predict_and_evaluate(x, y)
                test_predictions.append(predicted)
            print("Epoch ", epoch)
            print(" Train Acc: ", accuracy_score(y_train, train_predictions))
            print(" Test Acc: ", accuracy_score(y_test, test_predictions))