import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn import svm, metrics
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix

# Cargamos el conjunto de datos
digits = load_digits()

# Mostramos los dígitos en imágenes
fig = plt.figure(figsize=(6, 6))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

for i in range(10):
    ax = fig.add_subplot(2, 5, i + 1, xticks=[], yticks=[])
    ax.imshow(digits.images[i], cmap=plt.cm.binary, interpolation='nearest')
    ax.text(0, 7, str(digits.target[i]))

# Dividimos los datos en entrenamiento y prueba
Xtrain, Xtest, ytrain, ytest = train_test_split(digits.data, digits.target, random_state=0)

# Definimos la configuración del clasificador
clf = svm.SVC(kernel='rbf')

# Entrenamos el clasificador con los datos de entrenamiento
clf.fit(Xtrain, ytrain)

plt.show()

# Predicción
score = clf.score(Xtest, ytest)
print(score)

ypred = clf.predict(Xtest)
matriz = confusion_matrix(ytest, ypred)

plot_confusion_matrix(conf_mat=matriz, figsize=(6, 6), show_normed=False)
plt.tight_layout()
plt.show()