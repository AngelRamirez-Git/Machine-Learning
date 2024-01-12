# veamos el uso de kernels para el problema de clases linealmente no separables
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from sklearn import svm 
from sklearn.datasets import make_circles

# generamos 50 muestras con dos características, asociadas a dos clases
X, y = make_circles(50, factor=.2, noise=.2)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')

# creamos el modelo SVM para clasificacion con kernel lineal/rbf y entrenamos el  modelo
clf = svm.SVC(kernel='linear', C=100).fit(X,y)

# graficamos los datos en el espacio de características
cmap = matplotlib.colors.ListedColormap( [ 'r', 'b'] )
plt.scatter(X[:, 0],X[:, 1], c=y, s=40, cmap=cmap)

# creamos un mesh para evaluar la función de decisión
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy,xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = clf.decision_function(xy).reshape(XX.shape)

# graficamos el hiperplano y el margen
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,linestyles=['--','-','--'])

# graficamos los vectores soporte
ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100, facecolors='none',edgecolors='none')

Z_pred = clf.predict(np.c_[XX.ravel(), YY.ravel()])
Z_pred = Z_pred.reshape(XX.shape)

cmap = matplotlib.colors.ListedColormap( [ 'r', 'b' ] )
plt.pcolormesh(XX, YY, Z_pred, cmap = cmap, alpha=0.1)

plt.grid()
plt.show()

# Aplicamos una operación kernel gaussiano para separar las clases
# gamma controla el efecto kernel, si es muy pequeño el modelo se parece al lineal
gamma = 1
Xr = np.exp(-gamma*(X ** 2).sum(1))

#graficamos el espacio de caracteristicas mapeado por el kernel
ax = plt.subplot(projection='3d')
cmap = matplotlib.colors.ListedColormap( [ 'r', 'b' ] )
ax.scatter3D(X[:, 0],X[:,1], Xr, c=y, s=50, cmap=cmap)

plt.grid()
plt.show()
