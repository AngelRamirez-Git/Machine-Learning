# Ejemplo de Kernel PCA
from sklearn.datasets import make_moons
from sklearn.decomposition import KernelPCA
import matplotlib.pyplot as plt

X,Y = make_moons (n_samples= 100, random_state= 123)
XKPCA = KernelPCA (n_components= 2, kernel= "rbf", gamma= 15). fit_transform(X)

plt.figure(figsize = (16, 6))

#figure 1
plt.subplot (1, 2, 1)
plt.scatter (X[Y==0, 0],X[Y==0, 1],color = "red", alpha = 0.5)
plt.scatter (X[Y==1, 0],X[Y==1, 1],color = "blue", alpha = 0.5)

plt.xlabel ("$x_1$", fontsize = 16)
plt.ylabel ("$x_2$", fontsize = 16)

#figure 2
plt.subplot (1, 2, 2)
plt.scatter (XKPCA[Y==0, 0],XKPCA[Y==0, 1],color = "red", alpha = 0.5)
plt.scatter (XKPCA[Y==1, 0],XKPCA[Y==1, 1],color = "blue", alpha = 0.5)

plt.xlabel ("$x_1$", fontsize = 16)
plt.ylabel ("$x_2$", fontsize = 16)

plt.show()
