import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

train_cat = np.matrix(np.loadtxt('data/train_cat.txt', delimiter = ','))
train_grass = np.matrix(np.loadtxt('data/train_grass.txt', delimiter = ','))
print(train_cat.shape)
print(train_grass.shape)


mu_cat = np.mean(train_cat, axis=1)
sigma_cat = np.cov(train_cat,bias=True)
mu_grass = np.mean(train_grass, axis=1)
sigma_grass = np.cov(train_grass,bias=True)
pi0 = train_grass.size/(train_grass.size+train_cat.size)
pi1 = train_cat.size/(train_grass.size+train_cat.size)

print(mu_cat[:2])
print(mu_grass[:2])
print(sigma_cat[:2,:2])
print(sigma_grass[:2,:2])
print(pi0)
print(pi1)

Y = plt.imread('data/my_lion.jpg') / 255
print(Y.shape)
truth = plt.imread('data/truth.png') 
Y = np.asarray(Y)
truth = np.asarray(truth)
M,N = Y.shape
result = np.empty((M-8,N-8), dtype=float)
MAE = 0

def dec_rule(x, mu, sigma, prior):
	param1 = -0.5*(x-mu).transpose() @ np.linalg.inv(sigma) @ (x-mu)
	#print(param1.shape)
	param2 = np.log(prior)
	param3 = -0.5*np.log(np.linalg.det(sigma))
	return (param1 + param2 + param3)
for i in range(M-8):
	for j in range(N-8):
		block = Y[i:i+8, j:j+8] #
		x = block.flatten()
		x = x[:, None]
		value0 = dec_rule(x,mu_grass,sigma_grass,pi0)
		#print(value0)
		value1 = dec_rule(x,mu_cat,sigma_cat,pi1)
		#print(value1)
		result[i,j] = 0 if value0>value1 else 255 #can set the else to 1, and later mul matrix by 255 
		#print(result[i,j])
		MAE = MAE + abs(result[i,j] - truth[i,j])

MAE = MAE/result.size
print(MAE)
print(Y.shape)
print(result)
plt.imshow(result, cmap=plt.cm.gray)
#plt.imshow(result, cmap='Greys',  interpolation='nearest')
plt.show()