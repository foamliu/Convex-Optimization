# Import packages.
import cvxpy as cp
import numpy as np

m = 3
n = 2
one = np.ones((m,))
one_t = np.ones((n,)).transpose()
d = np.array([2, 1]).transpose()
s = np.array([[1, 1], [0, 1], [1, 0]])


# Define and solve the CVXPY problem.
x = cp.Variable((m, n))

prob = cp.Problem(cp.Maximize(cp.sum(cp.multiply(x, s))),
                  [0 <= x, x <= 1, one * x >= d, x * one_t <= 1])
prob.solve()

# Print result.
print("\nThe optimal value is", prob.value)
print("A solution x is")
print(x.value)
print("A dual solution is")
print(prob.constraints[0].dual_value)
