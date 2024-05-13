import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import pickle

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


sigma_cat_inv = np.linalg.inv(sigma_cat)
sigma_grass_inv = np.linalg.inv(sigma_grass)
log_det_sigma_cat = np.log(np.linalg.det(sigma_cat))
log_det_sigma_grass = np.log(np.linalg.det(sigma_grass))

def dec_rule(x):
  x_cat_hat = x - mu_cat
  x_cat_hat_t = x_cat_hat.transpose()
  x_grass_hat = x - mu_grass
  x_grass_hat_t = x_grass_hat.transpose()
  param1_cat = -0.5* x_cat_hat_t@ sigma_cat_inv @ x_cat_hat
  param1_grass = -0.5*x_grass_hat_t @ sigma_grass_inv @ x_grass_hat
  #print(param1.shape)
  param2_cat = -0.5*log_det_sigma_cat
  param2_grass = -0.5*log_det_sigma_grass
  return (param1_cat + param2_cat - param1_grass - param2_grass)

tou = np.geomspace(start=1e-100, stop=1e20, num=100)
prob_detection = []
prob_false_alarm = []	

for t in range(0,len(tou)):
	print(t)
	true_positives = 0
	false_positives = 0
	for i in range(M-8):
		for j in range(N-8):
			block = Y[i:i+8, j:j+8] #
			x = block.flatten()
			x = x[:, None]
			value = dec_rule(x)
			#result = 1 if value > np.log(tou[t]) else 0
			if(value > np.log(tou[t])):
				if(truth[i,j] == 1):
					true_positives += 1
				else:
					false_positives += 1
	prob_detection.append(true_positives/grnd_truth_positives)
	prob_false_alarm.append(false_positives/grnd_truth_negatives)

tou_bay = pi0/pi1 #bayesian tou
true_positives_bay = 0
false_positives_bay = 0
for i in range(M-8):
	for j in range(N-8):
		block = Y[i:i+8, j:j+8] #
		x = block.flatten()
		x = x[:, None]
		value = dec_rule(x)
		if(value > np.log(tou_bay)):
				if(truth[i,j] == 1):
					true_positives_bay += 1
				else:
					false_positives_bay += 1
prob_detection_bay = true_positives_bay/grnd_truth_positives
prob_false_alarm_bay = false_positives_bay/grnd_truth_negatives
prob_detection = np.asarray(prob_detection)
prob_false_alarm = np.asarray(prob_false_alarm)
plt.plot(prob_false_alarm,prob_detection,markersize=12)
plt.plot(prob_false_alarm_bay,prob_detection_bay,'ro')
plt.xlim([0,1])
plt.ylim([0,1])
plt.show()