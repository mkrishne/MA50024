import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
import csv

class0 = []
class1 = []

# Reading csv file for male data
with open("data/homework4_class0.txt", "r") as csv_file:
  reader = csv.reader(csv_file, delimiter=' ')
  for row in reader:
    row = [i for i in row if i != '']
    class0.append(list(np.float_(row)))
  class0 = np.array(class0)
csv_file.close()
print(class0.shape)

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
y = np.vstack((np.zeros((class0.shape[0],1)),np.ones((class1.shape[0],1))))
lambd = 0.0001

theta = cp.Variable((d,1))
log_likelihood = cp.sum(cp.multiply(y, X @ theta) - cp.logistic(X @ theta))
prob = cp.Problem(cp.Maximize(log_likelihood/N - lambd * cp.norm(theta, 2)))
prob.solve()
theta_cap_cvx = theta.value
print(f"theta_cap by cvx method :\n {theta_cap_cvx}")


plt.scatter(class0[:,0], class0[:,1], edgecolor ="blue", marker ="o")
plt.scatter(class1[:,0], class1[:,1], c ="red", marker =".")
line_x = np.linspace(0, 9, 1000)
line_y = -theta_cap_cvx[2] / theta_cap_cvx[1] - theta_cap_cvx[0] / theta_cap_cvx[1] * line_x
plt.scatter(line_x,line_y, c ="black", linewidths=0.1)
plt.show()


mu_class0 = np.mean(class0.T, axis=1)
sigma_class0 = np.cov(class0.T,bias=True)
mu_class1 = np.mean(class1.T, axis=1)
sigma_class1 = np.cov(class1.T,bias=True)
pi0 = class0.size/(class0.size+class1.size)
pi1 = class0.size/(class0.size+class1.size)


sigma_class0_inv = np.linalg.inv(sigma_class0)
sigma_class1_inv = np.linalg.inv(sigma_class1)
log_det_sigma_class0 = np.log(np.linalg.det(sigma_class0))
log_det_sigma_class1 = np.log(np.linalg.det(sigma_class1))

def dec_rule(x):
  x_class0_hat = x - mu_class0
  x_class0_hat_t = x_class0_hat.transpose()
  x_class1_hat = x - mu_class1
  x_class1_hat_t = x_class1_hat.transpose()
  param1_class0 = -0.5* x_class0_hat_t@ sigma_class0_inv @ x_class0_hat
  param1_class1 = -0.5*x_class1_hat_t @ sigma_class1_inv @ x_class1_hat
  param2_class0 = -0.5*log_det_sigma_class0
  param2_class1 = -0.5*log_det_sigma_class1
  return (param1_class0 + param2_class0 - param1_class1 - param2_class1)


X1 = np.linspace(-5, 10, 100)
X2 = np.linspace(-5, 10, 100) 
Z = np.zeros((100,100))

for i in range(0,len(X1)):
  for j in range(0,len(X2)):
      x = [X1[i], X2[j]]
      if(dec_rule(x) > 0):
        Z[i,j] = 1

print(Z)
[X, Y] = np.meshgrid(X1, X2)
fig, ax = plt.subplots(1, 1)
plt.scatter(class0[:,0], class0[:,1], edgecolor ="blue", marker ="o")
plt.scatter(class1[:,0], class1[:,1], c ="red", marker =".")
ax.contour(X, Y, Z)
plt.show()
