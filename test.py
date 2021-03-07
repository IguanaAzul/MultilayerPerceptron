import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from SimplePerceptron import Perceptron

iris = pd.read_csv("Iris.csv", index_col="Id")
encoder = OneHotEncoder(sparse=False)
inputs = np.array(iris[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]])
outputs = encoder.fit_transform(iris[["Species"]])

acc = list()
for i in range(100):
    x_train, x_test, y_train, y_test = train_test_split(inputs, outputs)
    perceptron = Perceptron(4, 3, lr=0.1, activation="step")
    acc.append(perceptron.train_and_evaluate(x_train, y_train, x_test, y_test, 100, verbose=False))
acc = np.array(acc)
print("Acurácia média usando ativação step: ", acc.mean(), " desvio padrão: ", acc.std())

acc = list()
for i in range(100):
    x_train, x_test, y_train, y_test = train_test_split(inputs, outputs)
    perceptron = Perceptron(4, 3, lr=0.1, activation="sigmoid")
    acc.append(perceptron.train_and_evaluate(x_train, y_train, x_test, y_test, 100, verbose=False))
acc = np.array(acc)
print("Acurácia média usando ativação sigmoid: ", acc.mean(), " desvio padrão: ", acc.std())
