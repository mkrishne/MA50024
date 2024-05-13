import numpy as np
import numpy.matlib as npm
import scipy.stats as stats
from scipy.linalg import fractional_matrix_power
import matplotlib.pyplot as plt


X = stats.multivariate_normal.rvs([0,0],[[2,1],[1,2]],1000)
x1 = np.arange(-5, 5, 0.01)
x2 = np.arange(-5, 5, 0.01)
X1, X2 = np.meshgrid(x1,x2)
Xpos = np.empty(X1.shape + (2,))
Xpos[:,:,0] = X1
Xpos[:,:,1] = X2
F = stats.multivariate_normal.pdf(Xpos,[0,0],[[2,1],[1,2]])
plt.scatter(X[:,0],X[:,1])
plt.contour(x1,x2,F)
plt.show()

#================================================================
X = np.random.multivariate_normal([0,0],[[1,0],[0,1]],5000)
x1 = np.arange(-2.5, 2.5, 0.01)
x2 = np.arange(-2.5, 2.5, 0.01)
X1, X2 = np.meshgrid(x1,x2)
Xpos = np.empty(X1.shape + (2,))
Xpos[:,:,0] = X1
Xpos[:,:,1] = X2
F = stats.multivariate_normal.pdf(Xpos,[0,0],[[1,0],[0,1]])
plt.scatter(X[:,0],X[:,1])
plt.contour(x1,x2,F)
plt.show()

#================================================================
X = np.random.multivariate_normal([0,0],[[1,0],[0,1]],5000)
mu = np.array([0,0])
Sigma = np.array([[-2,1],[1,-2]])
S, U = np.linalg.eig(Sigma)  #Sigma = USU.T
s = np.diag(S)
A = np.matmul(U,fractional_matrix_power(s,0.5))
print(np.matmul(A,A.T))  #This should be equal to sigma

Sigma_half = fractional_matrix_power(Sigma,0.5)
Y = np.dot(Sigma_half, X.T) + npm.repmat(mu,5000,1).T
x1 = np.arange(-1, 5, 0.01)
x2 = np.arange(0, 10, 0.01)
X1, X2 = np.meshgrid(x1,x2)
Xpos = np.empty(X1.shape + (2,))
Xpos[:,:,0] = X1
Xpos[:,:,1] = X2
F = stats.multivariate_normal.pdf(Xpos,[2,6],[[2,1],[1,2]])
plt.scatter(Y.T[:,0],Y.T[:,1])
plt.contour(x1,x2,F)
plt.show()







