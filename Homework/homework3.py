# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.sparse import spdiags
import scipy
import numpy.matlib
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
A = np.diag(1/(2*dt) * np.ones(N-1), k=-1) + np.diag(1/(2*dt) * np.ones(N-1), k=1)
#Set the last value of the first row to 1/(2*dt)
A[0, -1] = 1/(2*dt)
#Set the first value of the last row to 1/(2*dt)
A[-1, 0] = 1/(2*dt)

A = 0.5 * A # Because the PDE is 0.5u_x

y0 = f(x)

print(A)

# %%
# Question 1 Part A
def advectionPDE(t, x, A):
   u_t = A @ x # u_t = 0.5 u_x
   return u_t

sol = solve_ivp(lambda t,x: advectionPDE(t, x, A), [0, term], y0, t_eval=np.arange(0, term + 0.5, 0.5))

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

# %%
# Question 1 Part B

def advectionPDE(t, x, A):
   # u_t = (1 + 2sin(5t) - H(x - 4)) u_x = u_x + 2sin(5t) u_x - H(x - 4) u_x
   u_x  = np.copy(A) # u_x term
   # make the first four rows of A 0 for the H(x - 4) term
   A[0:4, :] = 0
   u_t = (u_x + 2 * np.sin(5 * t) * u_x - A) @ x 
   return u_t

sol = solve_ivp(lambda t,x: advectionPDE(t, x, A), [0, term], y0, t_eval=np.arange(0, term + 0.5, 0.5))

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
# %%
# Question 2 Common
h = 0.5 # our x step size
# Set up the matrix A
m = 4 # N value in x and y directions
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

# %%
#Set up matrix B
e1 = np.ones(n) # vector of ones

#place the 1's in the correct places (-(n**2 - n), -n, n, (n**2 - n))
B = scipy.sparse.spdiags([e1, -1 * e1, e1, -1 * e1],
                         [-(n-m), -m, m, (n-m)], n, n)
B = 1/(2 * h) * scipy.sparse.csr_matrix(B)

# %%
#Set up matrix C 
Low1 = np.matlib.repmat(np.concatenate((np.ones((m-1)), np.array([0]))), 1,m).reshape(n)
Low2 = np.matlib.repmat(np.concatenate((np.array([1]),np.zeros((m-1)))), 1,m).reshape(n)
Up1 = np.roll(Low2, -1)
Up2 = np.roll(Low1, -m+1)
C = scipy.sparse.spdiags([Low2, -1 * Low1, Up2, -1 * Up1],
                         [-(m - 1), -1, 1, (m - 1)], n, n)
C = 1/(2 * h) * scipy.sparse.csr_matrix(C)
# %%
# Question 2 Part B
f = lambda x,y: np.exp(-2*x**2 - (y**2 / 20))
N = 64
L = 10
dt = (2 * L) / N
nu = 0.001
term = 4
x = np.arange(-L, L, dt)
y = np.arange(-L, L, dt)

X, Y = np.meshgrid(x, y)
y0 = f(X, Y)

y0 = y0.reshape(N**2)

# %%
def myODEFun(t, omega):
   #Solve for psi vector
   psi = scipy.sparse.linalg.spsolve(A, omega)
   # matrixone = -(C @ psi)
   # matrixtwo = (B @ omega)
   # matrixthree = matrixone @ matrixtwo
   # matrixfour = (B @ psi)
   # matrixfive = (C @ omega)
   # matricesix = matrixfour @ matrixfive
   # matrixseven = nu * A @ omega
   # matrixeight = matrixthree + matricesix + matrixseven
   #now we solve for omega_t
   omega_t = -(C @ psi) * (B @ omega) + (B @ psi) * (C @ omega) + nu * A @ omega

   return omega_t

#%%
sol = solve_ivp(lambda t, omega: myODEFun(t, omega), [0, term], y0, t_eval=np.arange(0, term + 0.5, 0.5))
X, T = np.meshgrid(x,sol.t)
fig, ax = plt.subplots(subplot_kw={"projection": "3d"},figsize =(25, 10))
surf = ax.plot_surface(X, T, sol.y.T,cmap='magma')
ax.plot3D(x, 0*x, f(x),'-r',linewidth=5)
plt.xlabel('x')
plt.ylabel('time')
plt.show()
# %%
