# Import packages.
import cvxpy as cp
import numpy as np

m = 20000
n = 20
one = np.ones((m,))
one_t = np.ones((n,)).transpose()
d = np.random.randint(low=1000, high=2000, size=(n,)).transpose()
print('d: ' + str(d))
s = np.random.randint(low=0, high=2, size=(m, n))
print('s: ' + str(s))
p = np.random.random(size=(n,))
print('p: ' + str(p))

# Define and solve the CVXPY problem.
x = cp.Variable((m, n))
u = cp.Variable((n,)).T

objective = cp.Minimize(p * u)
constraints = [0 <= x, 0 <= u, one * cp.multiply(x, s) + u >= d, cp.multiply(x, s) * one_t <= 1]
prob = cp.Problem(objective, constraints)
prob.solve(verbose=True)

# Print result.
print("\nThe optimal value is", prob.value)
print("A solution x is")
print(x.value)
print("A dual solution is")
print(constraints[2].dual_value)
print(constraints[3].dual_value)
