# Import packages.
import numpy as np
from cvxpy import *

m = 3
n = 2
one = np.ones((m,))
one_t = np.ones((n,)).transpose()
d = np.array([2, 1]).transpose()
s = np.array([[1, 1], [0, 1], [1, 0]])

# Define and solve the CVXPY problem.
x = Variable((m, n))
revenue = sum(multiply(x, s))
obj = Maximize(revenue)
constr = [0 <= x, x <= 1, one * x >= d, x * one_t <= 1]
prob = Problem(obj, constr)
opt_val = prob.solve()

# Print result.
print("\nThe optimal value is", opt_val)
print("A solution x is")
solution = x.value
print(solution)
print("A dual solution is")
print(prob.constraints[2].dual_value)
print(prob.constraints[3].dual_value)
