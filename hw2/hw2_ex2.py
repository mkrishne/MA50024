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

#least squares 
N = male_rows.shape[0] + female_rows.shape[0]
d = 3
x = np.vstack((male_rows[:,1:3],female_rows[:,1:3]))
X = np.column_stack((x, np.ones(N)))  #consider the basis function as 1 + x1 + x2
y = np.vstack((np.ones((male_rows.shape[0],1)),-1*np.ones((female_rows.shape[0],1))))
XtX = np.dot(X.T, X)
Xty = np.dot(X.T, y)
theta_cap = np.dot(np.linalg.pinv(XtX), Xty )
print(f"theta_cap by analytic method : {theta_cap}")

#male test
N_test_male = male_test_rows.shape[0]
x_male = male_test_rows[:,1:3]
X_male_test = np.column_stack((x_male, np.ones(N_test_male)))  #consider the basis function as 1 + x1 + x2
y_male_test = X_male_test@theta_cap
y_male_test = np.sign(y_male_test)
correct_prediction = y_male_test[np.where(y_male_test==1)]
print(f"predicted {correct_prediction.shape[0]} correctly out of {y_male_test.shape[0]-1} male samples")

#female test
N_test_female = female_test_rows.shape[0]
x_female = female_test_rows[:,1:3]
X_female_test = np.column_stack((x_female, np.ones(N_test_female)))  #consider the basis function as 1 + x1 + x2
y_female_test = X_female_test@theta_cap
y_female_test = np.sign(y_female_test)
correct_prediction = y_female_test[np.where(y_female_test==-1)]
print(f"predicted {correct_prediction.shape[0]} correctly out of {y_female_test.shape[0]-1} female samples")


# ===== CVX method ==========
theta = cp.Variable(d)
y_cvx = cp.reshape(y, (y.shape[0],))
cost = cp.Minimize(cp.sum_squares(X@theta-y_cvx))
prob = cp.Problem(cost)
prob.solve()
theta_cap_cvx = theta.value
print(f"theta_cap by cvx method : {theta_cap_cvx}")
