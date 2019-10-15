# Import packages.
import numpy as np
from cvxpy import *

m = 3
n = 2
one = np.ones((m,))
one_t = np.ones((n,)).transpose()
d = np.array([2, 2]).transpose()
s = np.array([[1, 0], [1, 0], [1, 1]])
p = np.array([1, 0.5])

# Define and solve the CVXPY problem.
x = Variable((m, n))
u = Variable((n,)).T

cost = p * u
obj = Minimize(cost)
constr = [0 <= x, 0 <= u, one * multiply(x, s) + u >= d, multiply(x, s) * one_t <= 1]
prob = Problem(obj, constr)
opt_val = prob.solve()

# Print result.
print("\nThe optimal value is", opt_val)
print("A solution x is")
print(x.value)
print("A dual solution is")
print(constr[2].dual_value)
print(constr[3].dual_value)
