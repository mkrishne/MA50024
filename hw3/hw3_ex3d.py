import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import cvxpy as cp

train_cat = np.matrix(np.loadtxt('data/train_cat.txt', delimiter = ','))
train_grass = np.matrix(np.loadtxt('data/train_grass.txt', delimiter = ','))
print(train_cat.shape)
print(train_grass.shape)


mu_cat = np.mean(train_cat, axis=1)
sigma_cat = np.cov(train_cat,bias=True)
mu_grass = np.mean(train_grass, axis=1)
sigma_grass = np.cov(train_grass,bias=True)

Y = plt.imread('data/cat_grass.jpg') / 255
print(Y.shape)
truth = plt.imread('data/truth.png') 
truth = np.asarray(truth)
print(truth.shape)
grnd_truth_negatives = truth[np.where(truth==0)].shape[0]
grnd_truth_positives = truth[np.where(truth!=0)].shape[0]
print(grnd_truth_positives)
print(grnd_truth_negatives)
Y = np.asarray(Y)
M,N = Y.shape
result = np.empty((M-8,N-8), dtype=float)

train_cat = train_cat.transpose()
train_grass = train_grass.transpose()
N = train_cat.shape[0] + train_grass.shape[0]
print(N)
d = train_cat.shape[1] + 1
print(d)
x = np.vstack((train_cat,train_grass))
X = np.column_stack((x, np.ones(N)))  #consider the basis function as 1 + x1 + x2+..+x64
y = np.vstack((np.ones((train_cat.shape[0],1)),-1*np.ones((train_grass.shape[0],1))))
print(X.shape)

theta = cp.Variable(d)
y_cvx = cp.reshape(y, (y.shape[0],))
cost = cp.Minimize(cp.sum_squares(X@theta-y_cvx))
prob = cp.Problem(cost)
prob.solve()
theta_cap_cvx = theta.value
print(f"theta_cap by cvx method : {theta_cap_cvx}")
theta_cap_cvx = theta_cap_cvx[:, None]
theta_cap_cvx = theta_cap_cvx[:-1]
print(theta_cap_cvx.shape)

tou = np.geomspace(start=-1, stop=1, num=20)
prob_detection = []
prob_false_alarm = []

M,N = Y.shape
for t in range(0,len(tou)):
	print(t)
	true_positives = 0
	false_positives = 0
	for i in range(M-8):
		for j in range(N-8):
			block = Y[i:i+8, j:j+8] #
			x = block.flatten()
			x = x[:, None]
			#print(x.shape)
			value = theta_cap_cvx.transpose() @ x
			#result = 1 if value > np.log(tou[t]) else 0
			if(value > tou[t]):
				if(truth[i,j] == 1):
					true_positives += 1
				else:
					false_positives += 1
	prob_detection.append(true_positives/grnd_truth_positives)
	prob_false_alarm.append(false_positives/grnd_truth_negatives)

prob_detection = np.asarray(prob_detection)
prob_false_alarm = np.asarray(prob_false_alarm)
plt.plot(prob_false_alarm,prob_detection,markersize=12)
plt.show()
