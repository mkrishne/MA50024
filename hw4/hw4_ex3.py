import numpy as np
from numpy.matlib import repmat
import matplotlib.pyplot as plt
import cvxpy as cp
import csv

class0 = []
class1 = []

# Reading csv file for male data
#with open("data2/quiz4_class0.txt", "r") as csv_file:
with open("data/homework4_class0.txt", "r") as csv_file:
  reader = csv.reader(csv_file, delimiter=' ')
  for row in reader:
    row = [i for i in row if i != '']
    class0.append(list(np.float_(row)))
  class0 = np.array(class0)
csv_file.close()
print(class0.shape)

#with open("data2/quiz4_class1.txt", "r") as csv_file:
with open("data/homework4_class1.txt", "r") as csv_file:
  reader = csv.reader(csv_file, delimiter=' ')
  for row in reader:
    row = [i for i in row if i != '']
    class1.append(list(np.float_(row)))
  class1 = np.array(class1)
csv_file.close()
print(class1.shape)

#least squares 
N = class0.shape[0] + class1.shape[0]
d = 3
x = np.vstack((class0[:,0:2],class1[:,0:2]))
X = np.column_stack((x, np.ones(N)))  #consider the basis function as 1 + x1 + x2
print(X.shape)
y = np.vstack((np.zeros((class0.shape[0],1)),np.ones((class1.shape[0],1))))
#lambd = 0.0001
lambd = 0.01
one_transpose = np.ones((1,N))

K = np.zeros((100,100))
for i in range(0,N):
  for j in range(0,N):
    K[i,j] = np.exp(-np.sum((x[i]-x[j]) ** 2))

print(K[47:52,47:52])

alpha = cp.Variable((N,1))
log_likelihood = cp.sum(cp.multiply(y,K@alpha)) - cp.sum(cp.log_sum_exp(cp.hstack([np.zeros((N,1)),K@alpha]),axis=1))
prob = cp.Problem(cp.Maximize(log_likelihood/N - lambd * cp.quad_form(alpha, K)))
prob.solve()
alpha_cap_cvx = alpha.value
#print(alpha_cap_cvx)
print(f"first two values of alpha_cap by new cvx method :\n {alpha_cap_cvx[:2]}")

X1 = np.linspace(-5, 10, 100)
X2 = np.linspace(-5, 10, 100) 
Z = np.zeros((100,100))

for i in range(100):
  for j in range(100):
    data = repmat( np.array([X1[j], X2[i], 1]).reshape((1,3)), N, 1)
    s = data - X
    ks = np.exp(-np.sum(np.square(s), axis=1))
    Z[i,j] = np.dot(alpha_cap_cvx.T, ks).item()

plt.figure(figsize=(10,10))
plt.scatter(class0[:,0], class0[:,1], edgecolor ="blue", marker ="o")
plt.scatter(class1[:,0], class1[:,1], c ="red", marker =".")
plt.contour(X1, X2, Z>0, linewidths=1, colors='k')
plt.legend(['Class0', 'Class1'])
plt.title('Kernel Method')
plt.show()

'''
#following theta from alpha is done ony as an experiment; not part of hw
theta = alpha_cap_cvx.T@X
theta = theta[0]
print(theta)
plt.scatter(class0[:,0], class0[:,1], edgecolor ="blue", marker ="o")
plt.scatter(class1[:,0], class1[:,1], c ="red", marker =".")
line_x = np.linspace(-5, 9, 1000)
line_y = -theta[2]/theta[1] -theta[0] / theta[1] * line_x
plt.scatter(line_x,line_y, c ="black", linewidths=0.1)
plt.show()
'''
