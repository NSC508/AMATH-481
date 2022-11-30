# %%
import numpy as np
import scipy.sparse
import scipy.optimize
# %%
# Question 1
alpha = 2
L = 10
Time = 2
n = 128
xspan, dx = np.linspace(-L, L, n, endpoint=False, retstep=True)
tspan, dt = np.linspace(0, Time, 501, retstep=True)
lambda_star = (alpha * dt) / (dx**2)
# %%
e1 = -30 * np.ones(n)
e2 = 16 * np.ones(n)
e3 = -1 * np.ones(n)
A = scipy.sparse.spdiags([e3, e2, e1, e2, e3],
                         [-2, -1, 0, 1, 2], n, n)
A = scipy.sparse.lil_matrix(A)
A[0, -1] = 16
A[0, -2] = -1
A[1, -1] = -1
A[-1, 0] = 16
A[-1, 1] = -1
A[-2, 0] = -1

A = 1/12 * A
A3 = A.toarray()

# Solve
sol1 = np.zeros((len(xspan), len(tspan)))
u0 = 10 * np.cos(2 * np.pi * xspan / L) + 30 * np.cos(8 * np.pi * xspan / L)
sol1[:, 0] = u0
for i in range(len(tspan) - 1):
    u1 = u0 + lambda_star * (A@u0)
    sol1[:, i + 1] = u1 
    u0 = u1
A5 = sol1[:, -1]
print(A5)
# %%
# stability analysis
g = lambda z: 1 + 1/12 * lambda_star * (-30 + 32 * np.cos(z) - 2 * np.cos(2 * z))
A1 = abs(g(1))
minimum_index = scipy.optimize.fminbound(lambda z: -abs(g(z)), -np.pi, np.pi)
A2 = g(minimum_index)

# %%
# Question 2
e1 = -2 * np.ones(n)
e2 = np.ones(n)
mid = 1/2 * lambda_star * scipy.sparse.spdiags([e2, e1, e2],
                                              [-1, 0, 1], n, n)
mid = scipy.sparse.lil_matrix(mid)
mid[0, -1] = 1
mid[-1, 0] = 1
B = scipy.sparse.eye(n) - mid
C = scipy.sparse.eye(n) + mid
A7 = B.toarray()
A8 = C.toarray()
# %%
# Stability analysis
