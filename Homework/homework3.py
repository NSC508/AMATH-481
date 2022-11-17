# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.sparse import spdiags
import scipy
import numpy.matlib
import time
from matplotlib.animation import FuncAnimation
from random import randint
#import matplotlib.animation as animation 
# %% 
# Question 1 Common
f = lambda x: np.exp(-(x-5)**2) # our inital condition
L = 10 # our x boundary 
term = 10 # solve until time t = T 
N = 200 # the number of points in our x domain
dt = (2 * L)/N # our x step size
x = np.arange(-L, L, dt) # our x domain

#Set up the matrix A 
#it is a double-diagonal matrix with 1's on the lower and upper diagonals
#and 0's on the main diagonal
A = np.diag(-1 * np.ones(N-1), k=-1) + np.diag(np.ones(N-1), k=1)
#Set the last value of the first row to 1/(2*dt)
A[0, -1] = -1
#Set the first value of the last row to 1/(2*dt)
A[-1, 0] = 1

A = A * 1/(2*dt)

y0 = f(x)
print(A)

# %%
# Question 1 Part A
def advectionPDE(t, u):
   u_t = 0.5 * (A @ u) # u_t = 0.5 u_x
   return u_t

sol = solve_ivp(lambda t,u: advectionPDE(t, u), [0, term], y0, t_eval=np.arange(0, term + 0.5, 0.5))

print(sol.y)
# Create surface plot
X, T = np.meshgrid(x,sol.t)
fig, ax = plt.subplots(subplot_kw={"projection": "3d"},figsize =(25, 10))
surf = ax.plot_surface(X, T, sol.y.T,cmap='magma')
ax.plot3D(x, 0*x, f(x),'-r',linewidth=5)
plt.xlabel('x')
plt.ylabel('time')
# title = 'Advection PDE with c(t, x) = -0.5
plt.title('Advection PDE with c(t, x) = -0.5')
plt.show()

A1 = np.copy(A)
A2 = np.copy(sol.y)

# %%
# Question 1 Part B
c = lambda t, x: 1 + 2 * np.sin(5 * t) - np.heaviside(x - 4, 0)
def advectionPDE(t, u, A):
   # u_t = (1 + 2sin(5t) - H(x - 4)) u_x = u_x + 2sin(5t) u_x - H(x - 4) u_x
   # make the first four rows of A 0 for the H(x - 4) term
   u_t =  c(t, x) * (A @ u)
   return u_t

sol = solve_ivp(lambda t,u: advectionPDE(t, u, A), [0, term], y0, t_eval=np.arange(0, term + 0.5, 0.5))

# Create surface plot
X, T = np.meshgrid(x,sol.t)
fig, ax = plt.subplots(subplot_kw={"projection": "3d"},figsize =(25, 10))
surf = ax.plot_surface(X, T, sol.y.T,cmap='magma')
ax.plot3D(x, 0*x, f(x),'-r',linewidth=5)
plt.xlabel('x')
plt.ylabel('time')
# title = 'Advection PDE with c(t, x) = (1 + 2sin(5t) - H(x - 4))
plt.title('Advection PDE with c(t, x) = (1 + 2sin(5t) - H(x - 4))')
plt.show()

A3 = np.copy(sol.y)

print(A2)
# %%
# Question 2 Common
f = lambda x,y: np.exp(-2*x**2 - (y**2 / 20))
N = 64
L = 10
dt = (2 * L) / N
print(dt)
nu = 0.001
term = 4
x = np.arange(-L, L, dt)
y = np.arange(-L, L, dt)

X, Y = np.meshgrid(x, y)
y0 = f(X, Y)

y0 = np.transpose(y0).reshape(N**2)
h = 0.5 # our t step size

# %%
# Set up the matrix A
m = N # N value in x and y directions
n = m*m # total size of matrix

e1 = np.ones(n) # vector of ones
Low1 = np.tile(np.concatenate((np.ones(m-1), [0])), (m,)) # Lower diagonal 1
Low2 = np.tile(np.concatenate(([1], np.zeros(m-1))), (m,)) #Lower diagonal 2
                                    # Low2 is NOT on the second lower diagonal,
                                    # it is just the next lower diagonal we see
                                    # in the matrix.

Up1 = np.roll(Low1, 1) # Shift the array for spdiags
Up2 = np.roll(Low2, m-1) # Shift the other array

A = scipy.sparse.spdiags([e1, e1, Low2, Low1, -4*e1, Up1, Up2, e1, e1],
                         [-(n-m), -m, -m+1, -1, 0, 1, m-1, m, (n-m)], n, n)

#Set the first element of A equal to 2
A = scipy.sparse.csr_matrix(A)

A[0, 0] = 2
A = 1/(dt**2) * A
print(A.todense())
# %%
#Set up matrix B
e1 = np.ones(n) # vector of ones

#place the 1's in the correct places (-(n**2 - n), -n, n, (n**2 - n))
B = scipy.sparse.spdiags([e1, -1 * e1, e1, -1 * e1],
                         [-(n-m), -m, m, (n-m)], n, n)
B = 1/(2 * dt) * scipy.sparse.csr_matrix(B)

# %%
#Set up matrix C 
Low1 = np.matlib.repmat(np.concatenate((np.ones((m-1)), np.array([0]))), 1,m).reshape(n)
Low2 = np.matlib.repmat(np.concatenate((np.array([1]),np.zeros((m-1)))), 1,m).reshape(n)
Up1 = np.roll(Low2, -1)
Up2 = np.roll(Low1, -m+1)
C = scipy.sparse.spdiags([Low2, -1 * Low1, Up2, -1 * Up1],
                         [-(m - 1), -1, 1, (m - 1)], n, n)
print(C.todense())
C = 1/(2 * dt) * scipy.sparse.csr_matrix(C)
#print(C.todense())
# %%
def myODEFunGauss(t, omega):
   #Solve for psi vector
   psi = scipy.sparse.linalg.spsolve(A, omega)
   omega_t = (C @ psi) * (B @ omega) - (B @ psi) * (C @ omega) + nu * A @ omega
   return omega_t

LU = scipy.sparse.linalg.splu(A)
def myODEFunLU(t, omega):
   #Solve for psi vector using LU decomposition
   psi = LU.solve(omega)
   omega_t = (C @ psi) * (B @ omega) - (B @ psi) * (C @ omega) + (nu * (A @ omega))
   return omega_t

#%%
tic = time.time()
sol = solve_ivp(lambda t, omega: myODEFunGauss(t, omega), [0, term], y0, t_eval=np.arange(0, term + h, h))
toc = time.time()

print('Time to solve using Gaussian elimination: ' + str(toc - tic) + ' seconds')

tic = time.time()
sol2 = solve_ivp(lambda t, omega: myODEFunLU(t, omega), [0, term], y0, t_eval=np.arange(0, term + h, h))
toc = time.time()

print('Time to solve using LU decomposition: ' + str(toc - tic) + ' seconds')
# transpose the solution matrix
sol.y = np.transpose(sol.y)
sol2.y = np.transpose(sol2.y)

A4 = np.copy(A.todense())
A5 = np.copy(B.todense())
A6 = np.copy(C.todense())
A7 = np.copy(sol.y)
A8 = np.copy(sol2.y)

# %%
# split the A8 solution matrix of size 9 x 4096 into a matrix of size 9 x 64 x 64 and save it as A9
A9 = np.zeros((9, 64, 64)) 
for i in range(9):
   A9[i] = A8[i].reshape(64, 64)
# %%
# #split the  A7 solution matrix of size 9 x 4096 into a matrix of size 9 x 64 x 64 and save it as A10
# A10 = np.zeros((9, 64, 64))
# for i in range(9):
#    A10[i] = A7[i].reshape(64, 64)

# # plot each of the 9 solutions in a contourf plot in a 3 x 3 grid and hide the axis labels
# fig, axs = plt.subplots(3, 3, figsize=(5, 5))
# for i in range(3):
#    for j in range(3):
#       axs[i, j].contourf(Y, X, A9[i*3 + j])
#       axs[i, j].set_xticklabels([])
#       axs[i, j].set_yticklabels([])
#       axs[i, j].set_title('t = ' + str(i * 3 + j))
#       axs[i, j].set_xlabel('x')
#       axs[i, j].set_ylabel('y')
# plt.show()

# # plot each of the 9 solutions in a contourf plot in a 3 x 3 grid and hide the axis labels
# fig, axs = plt.subplots(3, 3, figsize=(5, 5))
# for i in range(3):
#    for j in range(3):
#       axs[i, j].contourf(Y, X, A10[i*3 + j])
#       axs[i, j].set_xticklabels([])
#       axs[i, j].set_yticklabels([])
#       axs[i, j].set_title('t = ' + str(i * 3 + j))
#       axs[i, j].set_xlabel('x')
#       axs[i, j].set_ylabel('y')
# plt.show()


# # %%
h = 0.01
sol3 = solve_ivp(lambda t, omega: myODEFunLU(t, omega), [0, term], y0, t_eval=np.arange(0, term + h, h))

solution = np.transpose(sol3.y)

#%%
#plot the solution in a contourf plot and animate it
fig = plt.figure()
ax = plt.axes(xlim=(-10, 10), ylim=(-10, 10))
contour = ax.contourf(Y, X, solution[0].reshape(64, 64))
fig.colorbar(contour, ax=ax)
def animate(i):
   contour = ax.contourf(Y, X, solution[i].reshape(64, 64))
   ax.set_title("Vortex Dynamics Equation Modeled at t = " + str(round(i * h, 2)))
   return contour
anim = FuncAnimation(fig, animate, frames=len(solution), interval=20, blit=False)
ax.set_xlabel('x')
ax.set_ylabel('y')
#set the title of the plot
plt.show()

anim.save('animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

# %%
