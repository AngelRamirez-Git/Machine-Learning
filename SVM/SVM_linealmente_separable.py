import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn import svm 
from sklearn.datasets import make_blobs

# generamos 30 muestras con dos características, asociadas a dos clases
X,y = make_blobs(n_samples=60, n_features=2, centers=2, random_state=21,center_box=(0,10.0))

# creamos el modelo SVM para clasificacion con kernel lineal y entrenamos el modelo
clf = svm.SVC(kernel='linear', C=100)
clf.fit(X, y)

# graficamos los datos en el espacio de características
cmap = matplotlib.colors.ListedColormap( [ 'purple', 'pink'] )
plt.scatter(X[:, 0],X[:, 1], c=y, s=40, cmap=cmap)

# creamos un mesh para evaluar la función de decisión
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = clf.decision_function(xy).reshape(XX.shape)

# graficamos el hiperplano y el margen
ax.contour(XX, YY, Z, colors ='k', levels=[-1, 0 ,1], alpha=0.5,linestyles=['--','-','--'])

# graficamos los vectores soporte
Z_pred = clf.predict(np.c_[XX.ravel(), YY.ravel()])
Z_pred = Z_pred.reshape(XX.shape)

cmap = matplotlib.colors.ListedColormap( [ 'pink', 'purple' ] )
plt.pcolormesh(XX, YY, Z_pred, cmap=cmap, alpha=0.1)

new_x = [[5,0.5]]
new_z = clf.predict(new_x)
if new_z[0]==0: color='black'
else: color='g'
plt.scatter(new_x[0][0],new_x[0][1], marker= '+',color=color,s=300)
plt.grid()
plt.show()
