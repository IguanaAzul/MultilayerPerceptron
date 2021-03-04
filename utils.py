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
        print("Available functions: ", function.keys())
        print("Using sigmoid")
        return sigmoid(x)
    return ni


def error_function(true, pred):
    true = np.array(true)
    pred = np.array(pred)
    return ((true - pred) ** 2).sum()


def d_error_function(true, pred):
    true = np.array(true)
    pred = np.array(pred)
    return 2 * (pred - true)


def dRelu(dError, out):
    dOut = np.array(dError, copy=True)
    dOut[out <= 0] = 0
    return dOut


def dSigmoid(dError, out):
    dOut = np.array(dError, copy=True)
    dOut[out <= 0] = 0
    return dOut


function = {"step": step, "sigmoid": sigmoid, "relu": relu}
d_funtion = {sigmoid: dSigmoid, relu: dRelu}
