# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.sparse import spdiags
# %%
# Question 1

#part a/b
N = 200
L = 10
dt = (2 * L)/N
print(dt)
x = np.arange(-L, L, dt)
c = lambda t, x: 0.5
f = lambda x: np.exp(-(x-5)**2)
#Set up the matrix A 

#it is a double-diagonal matrix with 1's on the lower and upper diagonals
#and 0's on the main diagonal
A = np.diag(np.ones(N-1), k=-1) + np.diag(np.ones(N-1), k=1)
#Set the last value of the first row to 1
A[0, -1] = 1
#Set the first value of the last row to 1
A[-1, 0] = 1

#multiple every value of A by 1/2(dt)
A = A * (1/(2*dt))

print(A)

A1 = A

sol = solve_ivp(lambda t, x: c(t, x) * A @ x, [0, L], y0 = f(x))
A2 = sol.y
# %%
