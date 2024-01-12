import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets, svm

# Generar datos de ejemplo no lineales
X, y = datasets.make_circles(n_samples=100, noise=0.05, random_state=42)

# Aplicar kernel RBF para transformar a un espacio tridimensional
r = np.exp(-(X ** 2).sum(1))

# Visualizar datos en el espacio tridimensional
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], r, c=y, s=50, cmap='viridis')
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('RBF Kernel Value')
ax.set_title('Transformación con Kernel RBF a Espacio 3D')

plt.show()
