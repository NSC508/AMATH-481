# %%
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
#%% 
n = 64
L = 2 * n
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

A_square = u*u + v*v
NL_u = lam(A_square) * u - omega(A_square) * v
NL_v = omega(A_square) * u - lam(A_square) * v

NL_u_hat = np.fft.fft2(NL_u)
NL_v_hat = np.fft.fft2(NL_v)

NL_u_hat = np.reshape(NL_u_hat, (n**2,1))
NL_v_hat = np.reshape(NL_v_hat, (n**2,1))

u0 = np.reshape(u, (n**2,1))
v0 = np.reshape(v, (n**2,1))

u_hat = NL_u_hat - D1 * kvec * u0
v_hat = NL_v_hat - D2 * kvec * v0

stacked = np.concatenate((u0,v0), axis=None)

A3 = np.real(u0)
A4 = np.imag(stacked)
# %%
#use RK45 to get the solution at each time step
def f(t, z, k):
    u = z[:n**2]
    v = z[n**2:]

    u = np.reshape(u, (n,n), order = 'F')
    v = np.reshape(v, (n,n), order = 'F')

    u = np.fft.ifft(u)
    v = np.fft.ifft(v)

    A_square = u*u + v*v
    NL_u = lam(A_square) * u - omega(A_square) * v
    NL_v = omega(A_square) * u - lam(A_square) * v

    NL_u_hat = np.fft.fft2(NL_u)
    NL_v_hat = np.fft.fft2(NL_v)

    NL_u_hat = np.reshape(NL_u_hat, n**2, order = 'F')
    NL_v_hat = np.reshape(NL_v_hat, n**2, order = 'F')

    u_hat = np.reshape(u, n**2, order = 'F')
    v_hat = np.reshape(v, n**2, order = 'F')

    u_hat = NL_u_hat - D1 * k * u_hat
    v_hat = NL_v_hat - D2 * k * v_hat

    u_hat = np.reshape(u_hat.T, (n**2))
    v_hat = np.reshape(v_hat.T, (n**2))
    stacked = np.concatenate((u_hat,v_hat), axis=None)
    return stacked
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
plt.show()

# %%
m = 3
alpha = 1
n = 30
x = np.linspace(-10, 10, n)
y = np.linspace(-10, 10, n)

A = np.diag(-1 * np.ones(n-1), k=-1) + np.diag(np.ones(n-1), k=1)

# %%
