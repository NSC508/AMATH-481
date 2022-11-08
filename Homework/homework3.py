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
#x, dt = np.linspace(-L, L, N, retstep= True)
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
sol = solve_ivp(lambda t, x: c(t, x) * A @ x, [0, 10], y0 = f(x), t_eval = np.arange(0, 10 + 0.5, 0.5))
A2 = sol.y

#Graph the solution x, t in three dimensions
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X, T = np.meshgrid(x, sol.t)
ax.plot_surface(X, T, A2.T, cmap='viridis')
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('u')
plt.show()

# %%
#part c
c = lambda t, x: 1 + 2 * np.sin(5 * t) - np.heaviside(x - 4, 0)

sol = solve_ivp(lambda t, x: c(t, x) * A @ x, [0, 10], y0 = f(x), t_eval = np.arange(0, 10 + 0.5, 0.5))
A3 = sol.y

#Graph the solution x, t in three dimensions
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X, T = np.meshgrid(x, sol.t)
ax.plot_surface(X, T, A3.T, cmap='viridis')
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('u')
plt.show()

# %%
