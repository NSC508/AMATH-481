# %%
import numpy as np
import scipy.sparse
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

A = 1/12 * lambda_star * A
print(A.toarray())
A4 = A.toarray()
# %%
