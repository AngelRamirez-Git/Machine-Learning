import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier

# Datos y Contexto
clientes = pd.read_csv("D:\Python\KNN\creditos.csv")
print(clientes)

# Filtrado del dataframe
buenos = clientes[clientes["cumplio"]==1]
malos = clientes[clientes["cumplio"]==0]
print(buenos,malos)

# Gráfica: Pagadores vs Deudores
plt.scatter(buenos["edad"], buenos["credito"],
            marker="*",s=150, color="green", label="Si pagó (Clase: 1)")

plt.scatter(malos["edad"], malos["credito"],
            marker="*",s=150, color="saddlebrown", label="No pagó (Clase: 0)")

plt.xlabel("Monto del crédito")
plt.ylabel("Edad")
plt.legend(bbox_to_anchor=(1, 0.2))

# Preparación de los datos (Escalar)
datos = clientes[["edad", "credito"]]
clase = clientes["cumplio"]

escalador = preprocessing.MinMaxScaler()
datos = escalador.fit_transform(datos)
print(datos)

# Creación del Modelo KNN
# Valor de K
clasificador = KNeighborsClassifier(n_neighbors=3)
clasificador.fit(datos, clase)
# Imprimir información del modelo
print("El modelo KNN se ha ajustado con exito a los datos.")

# Nuevo Solicitante (Clasificación)
edad = 55
monto = 600000

# Escalar los datos del nuevo solicitante
solicitante = escalador.transform([[edad, monto]])
print("Clase",clasificador.predict(solicitante))
print("Probabilidades por clase", clasificador.predict_proba(solicitante))

# Calcular clase y probabilidades

# Código para graficar
plt.scatter(edad, monto, marker= "P", s=250, color="red", label="Solicitante")
plt.xlabel("Monto del crédito")
plt.ylabel("Edad")
plt.legend(bbox_to_anchor=(1, 0.3))
plt.show()

# Regiones de las Clases: Pagadores VS Deudores
# Datos sintéticos de todos los posibles solicitantes
creditos = np.array([np.arange(100000, 600010, 1000)]*43).reshape(1,-1)
edades = np.array([np.arange(18,61)]*501).reshape(1, -1)
todos = pd.DataFrame(np.stack((edades, creditos), axis=2)[0],
                     columns=["edad", "credito"])

# Escalar los datos
solicitantes = escalador.transform(todos)

# Predecir todas las clases
clases_resultantes = clasificador.predict(solicitantes)

# Código para graficar
buenos = todos[clases_resultantes==1]
malos = todos[clases_resultantes==0]
plt.scatter(buenos["edad"], buenos["credito"],
            marker="*",s=150, color="green", label="Si pagará (Clase: 1)")
plt.scatter(malos["edad"], malos["credito"],
            marker="*",s=150, color="saddlebrown", label="No pagará (Clase: 0)")
plt.xlabel("Monto del crédito")
plt.ylabel("Edad")
plt.legend(bbox_to_anchor=(1, 0.2))
plt.show()
