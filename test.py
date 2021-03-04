import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from SimplePerceptron import Perceptron


iris = pd.read_csv("Iris.csv", index_col="Id")
encoder = OneHotEncoder(sparse=False)
inputs = np.array(iris[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]])
outputs = encoder.fit_transform(iris[["Species"]])

# treinar 1000 perceptrons usando ativação step e pegar a acurácia média
acc = list()
for i in range(10):
    x_train, x_test, y_train, y_test = train_test_split(inputs, outputs)
    perceptron = Perceptron(4, 3, lr=0.1, activation="step")
    acc.append(perceptron.train_and_evaluate(x_train, y_train, x_test, y_test, 100, verbose=False))
acc = np.array(acc)
print("Acurária média usando ativação step: ", acc.mean(), " desvio padrão: ", acc.std())

# treinar 1000 perceptrons usando ativação sigmoid e pegar a acurácia média
acc = list()
for i in range(10):
    x_train, x_test, y_train, y_test = train_test_split(inputs, outputs)
    perceptron = Perceptron(4, 3, lr=0.1, activation="sigmoid")
    acc.append(perceptron.train_and_evaluate(x_train, y_train, x_test, y_test, 100, verbose=False))
acc = np.array(acc)
print("Acurária média usando ativação sigmoid: ", acc.mean(), " desvio padrão: ", acc.std())

# Acurária média usando ativação step:  0.6288947368421053  desvio padrão:  0.11483336114137273
# Acurária média usando ativação sigmoid:  0.9315526315789473  desvio padrão:  0.06802414737458148
# A acurácia do sigmoid ficou muito melhor porque ativação degrau é muito brusca,
# durante o treinamento da passos grandes demais, o que impede ele de atingir os mínimos.
