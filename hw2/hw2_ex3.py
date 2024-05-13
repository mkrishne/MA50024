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

# a) visualize the classifier. 
N = male_rows.shape[0] + female_rows.shape[0]
d = 3
x = np.vstack((male_rows[:,1:3],female_rows[:,1:3]))
X = np.column_stack((x, np.ones(N)))  #consider the basis function as 1 + x1 + x2
y = np.vstack((np.ones((male_rows.shape[0],1)),-1*np.ones((female_rows.shape[0],1))))
XtX = np.dot(X.T, X)
Xty = np.dot(X.T, y)
theta_cap = np.dot(np.linalg.pinv(XtX), Xty )
print(f"theta_cap by analytic method : {theta_cap}")
plt.scatter(male_rows[:,1], male_rows[:,2], edgecolor ="blue", marker ="o")
plt.scatter(female_rows[:,1], female_rows[:,2], c ="red", marker =".")
line_x = np.linspace(0, 9, 1000)
line_y = -theta_cap[2] / theta_cap[1] - theta_cap[0] / theta_cap[1] * line_x
plt.scatter(line_x,line_y, c ="black", linewidths=0.1)
plt.show()

# b) classification accuracy
N_test_male = male_test_rows.shape[0]
x_male = male_test_rows[:,1:3]
X_male_test = np.column_stack((x_male, np.ones(N_test_male)))  #consider the basis function as 1 + x1 + x2
y_male_test = X_male_test@theta_cap
y_male_test = np.sign(y_male_test)
wrong_prediction_male = y_male_test[np.where(y_male_test==-1)]
correct_prediction_male = y_male_test[np.where(y_male_test==1)]
wrong_prediction_male_pcnt = 100*wrong_prediction_male.shape[0]/(y_male_test.shape[0]-1)
print(f"Type 2 error = {wrong_prediction_male_pcnt}%")
#female test
N_test_female = female_test_rows.shape[0]
x_female = female_test_rows[:,1:3]
X_female_test = np.column_stack((x_female, np.ones(N_test_female)))  #consider the basis function as 1 + x1 + x2
y_female_test = X_female_test@theta_cap
y_female_test = np.sign(y_female_test)
wrong_prediction_female = y_female_test[np.where(y_female_test==1)]
correct_prediction_female = y_female_test[np.where(y_female_test==-1)]
wrong_prediction_female_pcnt = 100*wrong_prediction_female.shape[0]/(y_female_test.shape[0]-1)
print(f"Type 1 error = {wrong_prediction_female_pcnt}%")

TruePositive = correct_prediction_male.shape[0]
FalsePositive = wrong_prediction_female.shape[0]
FalseNegative = wrong_prediction_male.shape[0]

Precision = TruePositive/(TruePositive+FalsePositive)
Recall = TruePositive/(TruePositive+FalseNegative)
print(f"Precision = {Precision}")
print(f"Recall={Recall}")

