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
                         [-2, -1, 0, 1, 2], n, n, format='csc')
#A = scipy.sparse.lil_matrix(A)
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
#reshape A5 to a 128x1 matrix
A5 = A5.reshape(128,1)
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
                                              [-1, 0, 1], n, n, format='csc')
#mid = scipy.sparse.lil_matrix(mid)
mid[0, -1] = 1/2 * lambda_star
mid[-1, 0] = 1/2 * lambda_star
B = scipy.sparse.eye(n) - mid
C = scipy.sparse.eye(n) + mid
A7 = B.toarray()
A8 = C.toarray()
print(A8)
# %%
# Solve suing LU decomposition 
sol2 = np.zeros((len(xspan), len(tspan)))
u0 = 10 * np.cos(2 * np.pi * xspan / L) + 30 * np.cos(8 * np.pi * xspan / L)
sol2[:, 0] = u0
PLU = scipy.sparse.linalg.splu(B)
test = PLU.solve(C@u0)
for i in range(len(tspan) - 1):
    u1 = PLU.solve(C@u0)
    sol2[:, i + 1] = u1
    u0 = u1
A9 = sol2[:, -1]
# reshape A9 to a 128x1 matrix
A9 = A9.reshape(128,1)
print(A9)

A10 = 0 
# %%
# Stability analysis
g = lambda z: (1 + lambda_star * (np.cos(z) - 1)) / (1 - lambda_star * (np.cos(z) - 1))
minimum_index = scipy.optimize.fminbound(lambda z: -abs(g(z)), -np.pi, np.pi)
A6 = g(minimum_index)

# %%
# Question 3