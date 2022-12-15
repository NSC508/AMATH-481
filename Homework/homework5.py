# %%
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
# %%
def cheb(N):
    # N is the number of points in the interior.
    if N==0:
        D = 0
        x = 1
        return D, x
    vals = np.linspace(0, N, N+1)
    x = np.cos(np.pi*vals/N)
    x = x.reshape(-1, 1)
    c = np.ones(N-1)
    c = np.pad(c, (1,), constant_values = 2)
    c *= (-1)**vals
    c = c.reshape(-1, 1)
    X = np.tile(x, (1, N+1))
    dX = X-X.T                  
    D  = (c*(1/c.T))/(dX+(np.eye(N+1)))       #off-diagonal entries
    D  = D - np.diag(sum(D.T))                #diagonal entries

    return D, x
#%% 
n = 64
L = 20
x = np.linspace(-10, 10, n)
y = np.linspace(-10, 10, n)
beta = 1 
D1 = 0.1
D2 = 0.1
dt = 0.5
t_span = np.arange(0, 25 + dt, dt)
X, Y = np.meshgrid(x,y)
m = 3
alpha = 0
u = (np.tanh(np.sqrt(X**2+Y**2))-alpha)*np.cos(m*np.angle(X+1j*Y) - np.sqrt(X**2+Y**2))
v = (np.tanh(np.sqrt(X**2+Y**2))-alpha)*np.sin(m*np.angle(X+1j*Y) - np.sqrt(X**2+Y**2))
r1 = np.arange(0, n/2, 1)
r2 = np.arange(-n/2, 0, 1)
kx = (2 * np.pi / L) * np.concatenate((r1, r2))
ky = kx.copy()
[KX, KY] = np.meshgrid(kx, ky)
K = KX**2 + KY**2
kvec = K.reshape(n**2, order = 'F')
kvec = np.reshape(kvec, (1, n**2))
A1 = X
A2 = u
# %%
lam = lambda A: 1 - A
omega = lambda A: -beta * A
A_square = u*u + v*v
# %%
#reshape u andv into vectors, then stack them together

u_hat_0 = np.fft.fft2(u)
v_hat_0 = np.fft.fft2(v)

#Save the real part of u_hat_0 to A3
A3 = np.real(u_hat_0)

# Reshape u_hat_0 and v_hat_0 into vectors, then stack them together
u_hat_0 = np.reshape(u_hat_0, n**2, order = 'F')
v_hat_0 = np.reshape(v_hat_0, n**2, order = 'F')
stacked = np.concatenate((u_hat_0,v_hat_0), axis=None)
stacked = np.reshape(stacked, (n**2*2, 1), order = 'F')
#save the imaginary part of stacked to A4
A4 = np.imag(stacked)
# %%
#use RK45 to get the solution at each time step
def f(t, z, k):    
    u_hat = z[:n**2]
    v_hat = z[n**2:]
    
    u_hat = np.reshape(u_hat, (n,n), order='F')
    v_hat = np.reshape(v_hat, (n,n), order='F')
    
    u = np.fft.ifft2(u_hat)
    v = np.fft.ifft2(v_hat)
    
    A_squared = u*u+v*v
    
    NL_U = lam(A_squared)*u-omega(A_squared)*v
    NL_V = omega(A_squared)*u-lam(A_squared)*v

    NL_U_Hat = np.fft.fft2(NL_U)
    NL_V_Hat = np.fft.fft2(NL_V)
    
    NL_U_Hat = np.reshape(NL_U_Hat, n**2, order='F')
    NL_V_Hat = np.reshape(NL_V_Hat, n**2, order='F')
    
    u_hat = np.reshape(u_hat, n**2, order='F')
    v_hat = np.reshape(v_hat, n**2, order='F')
    
    u_hat = NL_U_Hat - D1*k*u_hat
    v_hat = NL_V_Hat - D2*k*v_hat
    
    u_hat = np.reshape(u_hat.T, (n**2,))
    v_hat = np.reshape(v_hat.T, (n**2,))
    uvp1_flat = np.concatenate((u_hat,v_hat))
    
    return uvp1_flat
# %%
stacked = np.reshape(stacked, (n**2*2))
sol = solve_ivp(lambda t, z: f(t, z, kvec), [0, 25], stacked, method='RK45', t_eval=t_span)

print(sol.t)
A5 = np.real(sol.y)
A6 = np.imag(sol.y)

# %%
t_wanted_index = 4
u_wanted = sol.y[:n**2, t_wanted_index]
A7 = u_wanted

#reshape A7 into a one dimensional array
A7 = np.reshape(A7, (len(A7), 1))

u_wanted = np.reshape(u_wanted.T, (n,n)).T
A8 = np.real(u_wanted)
# %%
A9 = np.fft.ifft2(u_wanted)

#plot A9 in real space
plt.imshow(np.real(A9))
#change x and y bounds to be from -10 to 8
# plt.xlim(20, 40)
# plt.ylim(20, 40)
plt.colorbar()
plt.show()

# %%
# Question 2
m = 2
alpha = 1
N = 30
L = 20
x = np.linspace(-10, 10, n)
y = np.linspace(-10, 10, n)
alpha = 1
D, x = cheb(N)
x = x.reshape(N + 1)
x = x*L/2
D = D / (L / 2)
D1 = 0.1
D2 = 0.1
beta = 1

D_2 = D @ D
D_2 = D_2[1:-1, 1:-1]
x2 = x[1:-1]
y2 = x2.copy()
X, Y = np.meshgrid(x2, y2)

A11 = Y
t_span = np.arange(0, 25 + dt, dt)

U = (np.tanh(np.sqrt(X**2+Y**2))-alpha)*np.cos(m*np.angle(X+1j*Y) - np.sqrt(X**2+Y**2))
V = (np.tanh(np.sqrt(X**2+Y**2))-alpha)*np.sin(m*np.angle(X+1j*Y) - np.sqrt(X**2+Y**2))

A12 = V

stacked = np.concatenate((U,V), axis=None)

A13 = stacked

I = np.eye(len(D_2))

Lap = np.kron(D_2, I) + np.kron(I, D_2)

A10 = Lap
# %%

u_hat_0 = np.reshape(U, (N-1)**2, order = 'F')
v_hat_0 = np.reshape(V, (N-1)**2, order = 'F')
stacked = np.concatenate((u_hat_0,v_hat_0), axis=None)

A13 = stacked

A13 = np.reshape(A13, ((N-1)**2*2, 1))
# %%
def rhs(t, x):
    U = x[:(N-1)**2]
    V = x[(N-1)**2:]

    # U = np.reshape(U, (N-1,N-1), order='F')
    # V = np.reshape(V, (N-1,N-1), order='F')

    A_squared = U*U+V*V

    U_hat = lam(A_squared)@U - omega(A_squared)@V + (D1 * Lap)@U
    V_hat = omega(A_squared)@U - lam(A_squared)@V + (D2 * Lap)@V

    U_hat = np.reshape(U_hat, (N-1)**2, order='F')
    V_hat = np.reshape(V_hat, (N-1)**2, order='F')

    U_hat = np.reshape(U_hat.T, (N-1)**2)
    V_hat = np.reshape(V_hat.T, (N-1)**2)
    x = np.concatenate((U_hat,V_hat))

    return x

# %%

sol = solve_ivp(lambda t, z: rhs(t, z), [0, 25], stacked, method='RK45', t_eval=t_span)

print(sol.t)
A14 = sol.y

# %%
t_wanted_index = 4
u_wanted = sol.y[:n**2, t_wanted_index]
A15 = u_wanted

#reshape u_wanted into two (n-1) by (n-1) arrays
U_2 = u_wanted[:(N-1)**2]
V_2 = u_wanted[(N-1)**2:]

U_2 = np.reshape(U_2, (N-1,N-1), order='F')
V_2 = np.reshape(V_2, (N-1,N-1), order='F')

#padd the matrix with 0s in the first and last row and column
U_2 = np.pad(U_2, ((1,1),(1,1)), 'constant', constant_values=(0,0))
V_2 = np.pad(V_2, ((1,1),(1,1)), 'constant', constant_values=(0,0))

A16 = V_2
# %%
plt.imshow(A16)
#change x and y bounds to be from -10 to 8
# plt.xlim(20, 40)
# plt.ylim(20, 40)
plt.colorbar()
plt.show()
# %%
