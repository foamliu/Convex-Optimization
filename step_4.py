# Import packages.
import cvxpy as cp
import numpy as np

m = 3
n = 2
one = np.ones((m,))
one_t = np.ones((n,)).transpose()
d = np.array([2, 2]).transpose()
s = np.array([[1, 0], [1, 0], [1, 1]])
p = np.array([1, 0.5])

# Define and solve the CVXPY problem.
x = cp.Variable((m, n))
u = cp.Variable((n,)).T

objective = cp.Minimize(p * u)
constraints = [0 <= x, 0 <= u, one * cp.multiply(x, s) + u >= d, cp.multiply(x, s) * one_t <= 1]
prob = cp.Problem(objective, constraints)
prob.solve()

# Print result.
print("\nThe optimal value is", prob.value)
print("A solution x is")
print(x.value)
print("A dual solution is")
print(constraints[2].dual_value)
print(constraints[3].dual_value)
