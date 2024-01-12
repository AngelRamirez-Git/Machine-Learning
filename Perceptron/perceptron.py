"Perceptron Completo"
# Importación de bibliotecas:
# Matplotlib, que se utiliza para crear gráficos
# NumPy que se utiliza para realizar operaciones numéricas eficientes.
import matplotlib.pyplot as plt
import numpy as np

# Definición de datos de entrada y etiquetas de clase
# Se establecen los datos de entrada en forma de una matriz llamada "personas"
# que contiene pares de valores que representan la edad y el ahorro de un grupo de personas.
# Además, se crea un arreglo llamado "clases" que asigna una etiqueta de clase 
# (0 para "Denegada" y 1 para "Aprobada") a cada persona en función de algún criterio.
personas = np.array([[0.3, 0.4], [0.4, 0.3],
                     [0.3, 0.2], [0.4, 0.1],
                     [0.5, 0.2], [0.4, 0.8],
                     [0.6, 0.8], [0.5, 0.6],
                     [0.7, 0.6], [0.8, 0.5]])

# Clases: aprobado=1, denegado=0
clases = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

# Se utiliza Matplotlib para crear una gráfica de dispersión que muestra los datos en un gráfico bidimensional.
# Las personas se dividen en dos clases, "Denegada" y "Aprobada", y se representan como puntos en el gráfico.
# Esto ayuda a visualizar cómo se distribuyen las personas en función de su edad y ahorro.
plt.figure(figsize=(7, 7))
plt.title("¿Tarjeta Platinum?", fontsize=20)
plt.scatter(personas[clases == 0].T[0],
            personas[clases == 0].T[1],
            marker="x", s=180, color="brown",
            linewidths=5, label="Denegada")
plt.scatter(personas[clases == 1].T[0],
            personas[clases == 1].T[1],
            marker="o", s=180, color="green",
            linewidths=5, label="Aprobada")
plt.xlabel("Edad", fontsize=15)
plt.ylabel("Ahorro", fontsize=15)
plt.legend(bbox_to_anchor=(1.3, 0.15))
plt.box(False)
plt.xlim((0, 1.01))
plt.ylim((0, 1.01))
plt.grid()
plt.show()

# Inicializar pesos y umbral b aleatoriamente
# Estos valores son los parámetros que el perceptrón ajustará durante el entrenamiento para realizar la clasificación.
pesos = np.random.uniform(-1, 1, size=2)
b = np.random.uniform(-1, 1)

# Función de activación
# Toma los pesos, una entrada "x" y el umbral "b".
# Esta función calcula la salida del perceptrón aplicando una función de activación simple
# si la suma ponderada de las entradas supera el umbral, devuelve 1, de lo contrario, devuelve 0.
def activacion(pesos, x, b):
    """
    Calcula la salida de una función de activación para el perceptrón.
    """
    z = pesos * x
    if z.sum() + b > 0:
        return 1
    else:
        return 0

# Calcular resultado de la función de activación
# Se utiliza la función de activación para calcular la salida del perceptrón para una entrada de prueba.
# El resultado se almacena en la variable "RESULTADO_ACTIVACION".
RESULTADO_ACTIVACION = activacion(pesos, [0.7, 0.9], b)

# Imprimir resultados
print("Valores aleatorios de los pesos:", pesos)
print("Valor aleatorio del umbral b:", b)
print("Resultado de la funcion de activacion:", RESULTADO_ACTIVACION)

# Entrenamiento del perceptrón
# Se inicia un bucle que representa el proceso de entrenamiento del perceptrón.
# El entrenamiento se realiza durante un número fijo de épocas.
# En cada época, se calcula el error total y se ajustan los pesos y el umbral para minimizar este error.
TASA_DE_APRENDIZAJE = 0.01
EPOCAS = 100

for epoca in range(EPOCAS):
    ERROR_TOTAL = 0
    for i, persona in enumerate(personas):
        PREDICCION = activacion(pesos, persona, b)
        error = clases[i] - PREDICCION
        ERROR_TOTAL += error**2
        pesos[0] += TASA_DE_APRENDIZAJE * persona[0] * error
        pesos[1] += TASA_DE_APRENDIZAJE * persona[1] * error
        b += TASA_DE_APRENDIZAJE * error
    print(ERROR_TOTAL, end=" ")

# Después de completar todas las épocas de entrenamiento,
# se realizauna activación de prueba.
# El resultado se almacena en "RESULTADO_PRUEBA", lo que nos permite verificar cómo clasifica el perceptrón a esta persona de prueba.
RESULTADO_PRUEBA = activacion(pesos, [0.3, 10], b)
print()
print("Resultado de la activacion de prueba:", RESULTADO_PRUEBA)
