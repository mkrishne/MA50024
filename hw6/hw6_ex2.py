import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from random import randint
import math

TOTAL_EXP_NUM = 100000
NUM_EACH_COIN_FLIP = 10
TOTAL_NUM_COINS = 1000
x_err = np.arange(0,0.55,0.05)
prob_v1 = list(range(x_err.shape[0]))
prob_vrand = list(range(x_err.shape[0]))
prob_vmin = list(range(x_err.shape[0]))
hoeffding_bound = 2*np.exp(-2*NUM_EACH_COIN_FLIP*np.square(x_err))
exp_arr = np.zeros((TOTAL_NUM_COINS,NUM_EACH_COIN_FLIP))
v1 = list(range(TOTAL_EXP_NUM))
v_rand = list(range(TOTAL_EXP_NUM))
v_min = list(range(TOTAL_EXP_NUM))


print(exp_arr.shape)

def run_experiment():
	for coin in range (0,TOTAL_NUM_COINS):
		for flip_num in range (0,NUM_EACH_COIN_FLIP):
			exp_arr[coin,flip_num] = randint(0, 1)

v_value = [0,0,0]
def get_proportions():
	exp1 = exp_arr[0,:]
	#print(exp1)
	v_value[0] = (NUM_EACH_COIN_FLIP-np.count_nonzero(exp1))/NUM_EACH_COIN_FLIP
	rand_coin = randint(0,TOTAL_NUM_COINS-1)
	exp_rand =  exp_arr[rand_coin,:]
	#print(rand_coin)
	#print(exp_rand)
	v_value[1] = (NUM_EACH_COIN_FLIP-np.count_nonzero(exp_rand))/NUM_EACH_COIN_FLIP
	tails_for_coin = np.count_nonzero(exp_arr,axis=1)
	min_heads = NUM_EACH_COIN_FLIP - tails_for_coin.max()
	v_value[2] = min_heads/NUM_EACH_COIN_FLIP
	#print(v_value)
	return v_value

for exp_num in range (0,TOTAL_EXP_NUM):
	print(exp_num)
	run_experiment()
	v_value = get_proportions()
	v1[exp_num] = v_value[0]
	v_rand[exp_num] = v_value[1]
	v_min[exp_num] = v_value[2]

v1 = np.array(v1)
v_rand = np.array(v_rand)
v_min = np.array(v_min)
bins=np.arange(0,1.2,0.1)


hist_v1, bin_count = np.histogram(v1, bins=11, range=(0,1))
pdf_v1 = hist_v1/sum(hist_v1)
cdf_v1 = np.cumsum(pdf_v1)
'''
plt.plot(bins[:-1], pdf_v1, color="red", label="PDF")
plt.show()
'''

hist_vrand,bin_count = np.histogram(v_rand, bins=11, range=(0,1))
pdf_vrand = hist_vrand/sum(hist_vrand)
cdf_vrand = np.cumsum(pdf_vrand)

hist_vmin, bin_count = np.histogram(v_min, bins=11, range=(0,1))
pdf_vmin = hist_vmin/sum(hist_vmin)
cdf_vmin = np.cumsum(pdf_vmin)


plt.hist(v1, bins=11, range=(0,1))
plt.show()
plt.hist(v_rand, bins=11, range=(0,1))
plt.show()
plt.hist(v_min, bins=11, range=(0,1))
plt.show()


for i in range (x_err.shape[0]):
	low_bound = math.ceil((0.5-x_err[i])*10-1)
	up_bound = math.floor((0.5+x_err[i])*10+1)
	if low_bound < 0:
		low_bound = 0
	if up_bound > 10:
		up_bound = 10
	prob_v1[i] = cdf_v1[low_bound] + (1-cdf_v1[up_bound])
	prob_vrand[i] = cdf_vrand[low_bound] + (1-cdf_vrand[up_bound])
	prob_vmin[i] = cdf_vmin[low_bound] + (1-cdf_vmin[up_bound])


plt.plot(x_err,prob_v1, color="blue")
plt.plot(x_err,prob_vrand, color="green")
plt.plot(x_err,prob_vmin, color="orange")
plt.plot(x_err,hoeffding_bound, color="red")
plt.show()




