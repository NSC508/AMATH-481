# %%
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
#%% 
n = 64
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
A1 = X
A2 = u
# %%
u0 = np.fft.fft2(u)
v0 = np.fft.fft2(v)

A3 = np.real(u0)

#reshape u andv into vectors, then stack them together
u0 = np.reshape(u0, (n**2,1))
v0 = np.reshape(v0, (n**2,1))
stacked = np.vstack((u0,v0))

A4 = np.imag(stacked)
# %%
#use RK45 to get the solution at each time step
def f(t, z):
    z = np.reshape(z, (n**2*2))
    u = z[:n**2]
    v = z[n**2:]
    u = np.reshape(u, (n,n))
    v = np.reshape(v, (n,n))
    u_hat = np.fft.fft2(u)
    v_hat = np.fft.fft2(v)
    u_hat = np.reshape(u_hat, (n**2,1))
    v_hat = np.reshape(v_hat, (n**2,1))
    u_hat = np.vstack((u_hat,v_hat))
    return u_hat.T
# %%
stacked = np.reshape(stacked, (n**2*2))
sol = solve_ivp(f, [0, 25], stacked, method='RK45', t_eval=t_span)

print(sol.y.shape)
A5 = np.real(sol.y)
A6 = np.imag(sol.y)

# %%
t_wanted_index = 5
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
