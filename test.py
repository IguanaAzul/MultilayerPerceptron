import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from MultiLayerPerceptron import Network
from sklearn.model_selection import train_test_split
from SimplePerceptron import Perceptron


iris = pd.read_csv("Iris.csv", index_col="Id")
encoder = OneHotEncoder(sparse=False)
inputs = np.array(iris[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]])
outputs = encoder.fit_transform(iris[["Species"]])
x_train, x_test, y_train, y_test = train_test_split(inputs, outputs)

# treinar 100 perceptrons usando ativação step e pegar a acurácia média
acc = list()
for i in range(100):
    perceptron = Perceptron(4, 3, lr=0.1, activation="step")
    acc.append(perceptron.train_and_evaluate(x_train, y_train, x_test, y_test, 100, verbose=False))
acc = np.array(acc)
print("Acurária média usando ativação step: ", acc.mean(), " desvio padrão: ", acc.std())

# treinar 100 perceptrons usando ativação sigmoid e pegar a acurácia média
acc = list()
for i in range(100):
    perceptron = Perceptron(4, 3, lr=0.1, activation="sigmoid")
    acc.append(perceptron.train_and_evaluate(x_train, y_train, x_test, y_test, 100, verbose=False))
acc = np.array(acc)
print("Acurária média usando ativação sigmoid: ", acc.mean(), " desvio padrão: ", acc.std())
