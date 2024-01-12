"Perceptron utilizando Sklearn"
from sklearn.linear_model import Perceptron
import numpy as np

personas = np.array([[0.3, 0.4], [0.4, 0.3],
                     [0.3, 0.2], [0.4, 0.1],
                     [0.5, 0.2], [0.4, 0.8],
                     [0.6, 0.8], [0.5, 0.6],
                     [0.7, 0.6], [0.8, 0.5]])

# Clases: aprobado=1, denegado=0
clases = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

perceptron = Perceptron().fit(personas, clases)
predicciones = perceptron.predict([[0.2,0.2],[0.8,0.8]])

# Imprimir las predicciones
print("Predicciones para [[0.2, 0.2], [0.8, 0.8]]:")
print(predicciones)

