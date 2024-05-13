import numpy as np
from matplotlib import pyplot as plt
from scipy.special import eval_legendre
from scipy.optimize import linprog

x = np.linspace(-1,1,50)
beta = np.array([-0.001, 0.01, 0.55, 1.5, 1.2])
y = beta[0]*eval_legendre(0,x) + beta[1]*eval_legendre(1,x) + beta[2]*eval_legendre(2,x) + \
	beta[3]*eval_legendre(3,x) + beta[4]*eval_legendre(4,x) + np.random.normal(0, 0.2, 50)
X = np.column_stack((eval_legendre(0,x), eval_legendre(1,x), eval_legendre(2,x), \
	eval_legendre(3,x), eval_legendre(4,x)))
beta_hat = np.linalg.lstsq(X, y, rcond=None)[0]
print(beta_hat)
yhat = beta_hat[0]*eval_legendre(0,x) + beta_hat[1]*eval_legendre(1,x) + beta_hat[2]*eval_legendre(2,x) + \
		beta_hat[3]*eval_legendre(3,x) + beta_hat[4]*eval_legendre(4,x)
		
plt.plot(x,y,'o',markersize=12)
plt.plot(x,yhat, linewidth=8)
plt.savefig("3_a_n_c")
plt.show()
#================================================================================================
idx = [10, 16, 23, 37, 45];
y[idx] = 5;
idx = [5, 6];
y[idx] = 3;
beta_hat_with_outlier = np.linalg.lstsq(X, y, rcond=None)[0]
yhat_with_outlier = beta_hat_with_outlier[0]*eval_legendre(0,x) + beta_hat_with_outlier[1]*eval_legendre(1,x) + beta_hat_with_outlier[2]*eval_legendre(2,x) \
					+ beta_hat_with_outlier[3]*eval_legendre(3,x) + beta_hat_with_outlier[4]*eval_legendre(4,x)

plt.plot(x,y,'o',markersize=12)
plt.plot(x,yhat_with_outlier, 'r', linewidth=8)
plt.savefig("3_d")
plt.show()
#================================================================================================
X_lp = np.column_stack((eval_legendre(0,x), eval_legendre(1,x), eval_legendre(2,x), \
	eval_legendre(3,x), eval_legendre(4,x)))
A = np.vstack((np.hstack((X_lp, -np.eye(50))), np.hstack((-X_lp, -np.eye(50)))))
b = np.hstack((y,-y))
c = np.hstack((np.zeros(5), np.ones(50)))
res = linprog(c, A, b, bounds=(None,None), method="highs")
beta_cap_lp = res.x
yhat_lp = beta_cap_lp[0]*eval_legendre(0,x) + beta_cap_lp[1]*eval_legendre(1,x) + beta_cap_lp[2]*eval_legendre(2,x) + \
		beta_cap_lp[3]*eval_legendre(3,x) + beta_cap_lp[4]*eval_legendre(4,x)
plt.plot(x,y,'o',markersize=12)
plt.plot(x,yhat_lp, 'y', linewidth=8)
plt.savefig("3_d")
plt.show()
