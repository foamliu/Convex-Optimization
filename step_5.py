# Import packages.
import numpy as np
from cvxpy import *

m = 20000
n = 10
one = np.ones((m,))
one_t = np.ones((n,)).transpose()

# Generate random data.
d = np.random.randint(low=1000, high=2000, size=(n,)).transpose()
print('d: ' + str(d))
s = np.random.randint(low=0, high=2, size=(m, n))
print('s: ' + str(s))
p = np.random.random(size=(n,))
print('p: ' + str(p))

# Define and solve the CVXPY problem.
x = Variable((m, n))
u = Variable((n,)).T

cost = p * u
obj = Minimize(cost)
constr = [0 <= x, 0 <= u, one * multiply(x, s) + u >= d, multiply(x, s) * one_t <= 1]
prob = Problem(obj, constr)
opt_val = prob.solve(verbose=True)

# Print result.
print("\nThe optimal value is", opt_val)
print("A solution x is")
print(x.value)
print("A dual solution is")
print(constr[2].dual_value)
print(constr[3].dual_value)
