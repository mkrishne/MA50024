import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

x = np.linspace(-3,3,1000)
fx = stats.norm.pdf(x, 0, 1)
#plt.plot(x,fx,markersize=12)
#plt.show()
#========================================================

x_samp = np.random.normal(0,1,1000)
#plt.hist(x_samp,bins=4,density=True);
mu, sigma = stats.norm.fit(x_samp)
mean_n_sigma_string = str(mu) + " & " + str(sigma) 
print("mean and sd values of fit data are : " + mean_n_sigma_string + " respectively" )

fx = stats.norm.pdf(x, mu, sigma)
#plt.plot(x,fx,markersize=12)
#plt.show()

#plt.hist(x_samp,bins=1000,density=True);
#plt.plot(x,fx,markersize=12)
#plt.show()

#========================================================
n = 1000
x_samp = np.random.normal(0,1,n)
m = np.arange(1,201)
J = np.zeros((200))
max_value = np.max(x_samp)
min_value = np.min(x_samp)
for i in range(0,200):
	hist,bins = np.histogram(x_samp,bins=m[i])
	h = (max_value-min_value)/m[i]
	J[i] = 2/((n-1)*h)-((n+1)/((n-1)*h))*np.sum((hist/n)**2)
#plt.plot(m,J);
#plt.show()

min_J_value = np.min(J)
m_star = 1+np.where(J==min_J_value)[0][0]
print("m* that minimizes the risk value : " + str(m_star) )
plt.hist(x_samp,bins=m_star,density=True);
plt.plot(x,fx,markersize=12)
#plt.show()