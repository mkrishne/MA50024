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


def delta_f(theta_var):
	XtX = np.dot(X.T, X)
	Xty = np.dot(X.T, y)
	return (-2*Xty + 2*XtX@theta_var)

def iter(theta_var):
	XtX = np.dot(X.T, X)
	del_f = delta_f(theta_var)
	numerator = del_f.T@del_f
	denominator = 2*del_f.T@XtX@del_f
	return numerator/denominator

N = male_rows.shape[0] + female_rows.shape[0]
d = 3
x = np.vstack((male_rows[:,1:3],female_rows[:,1:3]))
X = np.column_stack((x, np.ones(N)))  #consider the basis function as 1 + x1 + x2
y = np.vstack((np.ones((male_rows.shape[0],1)),-1*np.ones((female_rows.shape[0],1))))
theta_not = np.zeros((d,1))

N_test= male_test_rows.shape[0] + female_test_rows.shape[0]
x_test = np.vstack((male_test_rows[:,1:3],female_test_rows[:,1:3]))
X_test = np.column_stack((x_test, np.ones(N_test)))  #consider the basis function as 1 + x1 + x2
y_test = np.vstack((np.ones((male_test_rows.shape[0],1)),-1*np.ones((female_test_rows.shape[0],1))))
theta = theta_not
sq_err = []

x_axis = np.arange(0,50000)


for k in x_axis:
	theta = theta - iter(theta)*delta_f(theta)
	sq_err.append(np.sum((y_test - (X_test@theta))**2))
print(theta)
plt.semilogx(x_axis, sq_err, linewidth=8)
plt.grid(True)
plt.show()


#momentum method
sq_err_momentum = []
thetak = theta_not
thetak_minus1 = theta_not
beta = 0.9
for k in x_axis:
	curr_iter = iter(thetak)
	thetak_plus1 = thetak - beta*curr_iter*delta_f(thetak_minus1) - (1-beta)*curr_iter*delta_f(thetak)
	thetak_minus1 = thetak
	thetak = thetak_plus1
	sq_err_momentum.append(np.sum((y_test - (X_test@thetak_plus1))**2))
print(sq_err_momentum)
print(thetak_plus1)
plt.semilogx(x_axis, sq_err_momentum, linewidth=8)
plt.grid(True)
plt.show()