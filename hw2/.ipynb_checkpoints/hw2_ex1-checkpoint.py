import numpy as np
import matplotlib.pyplot as plt
from scipy.special import eval_legendre
import cvxpy as cvx
np.set_printoptions(precision=2, suppress=True)

# Setup the problem
N = 10
x = np.linspace(-1,1,N)
a = np.array([1, 0.5, 0.5, 1.5, 1])
y = a[0]*eval_legendre(0,x) + a[1]*eval_legendre(1,x) + \
    a[2]*eval_legendre(2,x) + a[3]*eval_legendre(3,x) + \
    a[4]*eval_legendre(4,x) + 0.25*np.random.randn(N)
print(x)
print(a)
print(y)
# Solve the regression problem
d = 10
theta_true = np.zeros(d)
theta_true[0:5] = a
print(theta_true)
X = np.zeros((N, d))
print(X)
for p in range(d):
  X[:,p] = eval_legendre(p,x)
print(X)

lambd = 0
theta       = cvx.Variable(d)
objective   = cvx.Minimize( cvx.sum_squares(X*theta-y) )
prob        = cvx.Problem(objective)
prob.solve()
theta_vanilla = theta.value
print(theta_vanilla)
lambd = 0.5
theta       = cvx.Variable(d)
objective   = cvx.Minimize( cvx.sum_squares(X*theta-y) + lambd*cvx.sum_squares(theta) )
prob        = cvx.Problem(objective)
prob.solve()
theta_ridge = theta.value

lambd = 1
theta       = cvx.Variable(d)
objective   = cvx.Minimize( cvx.sum_squares(X*theta-y) + lambd*cvx.norm1(theta) )
prob        = cvx.Problem(objective)
prob.solve()
theta_LASSO = theta.value