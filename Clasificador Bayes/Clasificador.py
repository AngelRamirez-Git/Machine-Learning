# Importación de bibliotecas
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, precision_score

# Importación de datos
dataset = datasets.load_breast_cancer()
x = dataset.data  # Todas las columnas
y = dataset.target  # Datos correspondientes a las etiquetas

# Mostrar información del dataset
print('Informacion en el dataset:')
print(dataset)
print("\n")

# Entender los datos
print('Claves del dataset:')
print(dataset.keys())
print("\n")

# Verificamos las características del dataset
print('Caracteristicas del dataset:')
print(dataset.DESCR)
print("\n")

# Separación de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Escalado de datos
escalar = StandardScaler()
X_train = escalar.fit_transform(X_train)
X_test = escalar.transform(X_test)

# Algoritmo Naive Bayes
algoritmo = GaussianNB()

# Entrenamiento del modelo
algoritmo.fit(X_train, y_train)

# Predicción y matriz de confusión
y_pred = algoritmo.predict(X_test)
matriz = confusion_matrix(y_test, y_pred)

# Cálculo de datos predichos correctamente e incorrectamente
datos_correctos = matriz[0, 0] + matriz[1, 1]
dato_erroneo = matriz[0, 1]

# Muestra de resultados
print("Resultados de la prediccion:")
print(f"Datos predichos correctamente: {datos_correctos}")
print(f"Dato erroneo obtenido: {dato_erroneo}")
print("\n")

# Cálculo de la precisión del modelo
precision = precision_score(y_test, y_pred)
print('Precision del modelo:')
print(precision)
