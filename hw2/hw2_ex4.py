import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
import csv

male_rows = []
female_rows = []
male_test_rows = []
female_test_rows = []

# Reading csv files for male data
with open("data/male_train_data.csv", "r") as csv_file:
  reader = csv.reader(csv_file, delimiter=',')
  fields = next(reader)
  for row in reader:
        male_rows.append(list(np.float_(row)))
  male_rows = np.array(male_rows)
  male_rows[:,1] = np.divide(male_rows[:,1],10)
  male_rows[:,2] = np.divide(male_rows[:,2],1000)
csv_file.close()

with open("data/male_test_data.csv", "r") as csv_file:
  reader = csv.reader(csv_file, delimiter=',')
  fields = next(reader)
  for row in reader:
        male_test_rows.append(list(np.float_(row)))
  male_test_rows = np.array(male_test_rows)
  male_test_rows[:,1] = np.divide(male_test_rows[:,1],10)
  male_test_rows[:,2] = np.divide(male_test_rows[:,2],1000)
csv_file.close()

# Reading csv files for female data
with open("data/female_train_data.csv", "r") as csv_file:
  reader = csv.reader(csv_file, delimiter=',')
  fields = next(reader)
  for row in reader:
        female_rows.append(list(np.float_(row)))
  female_rows = np.array(female_rows)
  female_rows[:,1] = np.divide(female_rows[:,1],10)
  female_rows[:,2] = np.divide(female_rows[:,2],1000)
csv_file.close()

with open("data/female_test_data.csv", "r") as csv_file:
  reader = csv.reader(csv_file, delimiter=',')
  fields = next(reader)
  for row in reader:
        female_test_rows.append(list(np.float_(row)))
  female_test_rows = np.array(female_test_rows)
  female_test_rows[:,1] = np.divide(female_test_rows[:,1],10)
  female_test_rows[:,2] = np.divide(female_test_rows[:,2],1000)
csv_file.close()

N = male_rows.shape[0] + female_rows.shape[0]
d = 3
x = np.vstack((male_rows[:,1:3],female_rows[:,1:3]))
X = np.column_stack((x, np.ones(N)))  #consider the basis function as 1 + x1 + x2
y = np.vstack((np.ones((male_rows.shape[0],1)),-1*np.ones((female_rows.shape[0],1))))

lambd = np.arange(0.1, 10, 0.1)
theta_lambd_norm = np.zeros(len(lambd))
mse = np.zeros(len(lambd))


for idx, lam in enumerate(lambd):
    theta_lambd = cp.Variable((d,1))
    objective_lambd = cp.Minimize(cp.sum_squares(X@theta_lambd - y) + lam*cp.sum_squares(theta_lambd))
    prob = cp.Problem(objective_lambd)
    prob.solve()
    theta_solution = theta_lambd.value
    theta_lambd_norm[idx] = (np.linalg.norm(theta_solution))**2
    mse[idx] = (np.linalg.norm(np.matmul(X, theta_solution)-y))**2

plt.plot(theta_lambd_norm, mse, linewidth=8)
plt.grid(True)
plt.show()


plt.plot(lambd, mse, linewidth=8)
plt.grid(True)
plt.show()

plt.plot(lambd, theta_lambd_norm, linewidth=8)
plt.grid(True)
plt.show()




