# -*- coding: utf-8 -*-
"""RedNeuronal.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1GIdypBr2toeiX-D3H5XTqsV9VbgD8BfK
"""

import tensorflow as tf
import numpy as np

celsius = np.array([0, -6.66667, -9.44444, 10, 18.3333, 22.2222, -15, 29.4444, 37.7778, -17.7778], dtype=float)
fahrenheit = np.array([32, 20, 15, 50, 65, 72, 5, 85, 100, 0], dtype=float)

# capa = tf.keras.layers.Dense(units=1, input_shape=[1])
# modelo = tf.keras.Sequential([capa])
oculta1 = tf.keras.layers.Dense(units=3, input_shape=[1])
oculta2 = tf.keras.layers.Dense(units=3)
salida = tf.keras.layers.Dense(units=1)
modelo = tf.keras.Sequential([oculta1,oculta2, salida])
modelo.compile(
    optimizer=tf.keras.optimizers.Adam(0.1),
    loss='mean_squared_error'
)

print("Comenzando entrenamiento...")
historial = modelo.fit(celsius, fahrenheit, epochs=1000, verbose=False)
print("Modelo entrenado!")

import matplotlib.pyplot as plt
plt.xlabel("N Epoca")
plt.ylabel("Magnitud de pérdida")
plt.plot(historial.history["loss"])

print("Hagamos una predicción!")
resultado = modelo.predict([100.0])
print("El resultado es " + str(resultado) + " fahrenheit!")

print("Variables internas del modelo")
# print(capa.get_weights())
print(oculta1.get_weights())
print(oculta2.get_weights())
print(salida.get_weights())