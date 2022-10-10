# %%
import numpy as np
from numba import njit, jit

# Question 1 Part A

# %% 
dydt = lambda y, t: -3 * y + np.sin(t)
dt = np.arange(2, 9)
y0 = np.pi / np.sqrt(2)
t = np.arange(0, 6)
dt = 2**dt
dt = 1/dt

# %%
@jit
def forward_euler(dydt, y0, t, dt):
    y = np.zeros(1, len(t))
    y[0] = y0
    for i in range(1, len(t)):
        y[i] = y[i-1] + dydt(y[i-1], t[i-1]) * dt
    return y

# %%
for i in dt:
    res = forward_euler(dydt, y0, t, i)
    print(res)
# %%
