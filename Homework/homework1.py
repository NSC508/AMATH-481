# %%
import numpy as np
import matplotlib.pyplot as plt
import scipy
# %%
#Question 1 constants 
dydt = lambda y, t: -3 * y * np.sin(t)
dt = np.arange(2, 9)
dt = 2**dt
dt = 1/dt
y0 = np.pi / np.sqrt(2)
t_last = 5
ytrue = lambda t: (np.pi * (np.e ** (3 * (np.cos(t) - 1)))) / (np.sqrt(2))
# %%
# Question 1A
def forward_euler(dydt, y0, dt):
    t = np.arange(0, t_last + dt, dt)
    y = np.zeros(len(t))
    y[0] = y0
    for i in range(1, len(t)):
        y[i] = y[i-1] + dydt(y[i-1], t[i-1]) * dt
    return y

error_vals = []
A1 = []
for i in dt:
    t = np.arange(0, t_last + i, i)
    res = forward_euler(dydt, y0, i)
    error = np.abs(res[-1] - ytrue(t[-1]))
    error_vals.append(error)
    if i == (1 / (2 ** 8)):
        A1 = res
    
#plot log of dt vs log of error with title "Forward Euler Error" and np.polyfit for line of best fit
plt.loglog(dt, error_vals, 'o')
a, b = np.polyfit(np.log(dt), np.log(error_vals), 1)
plt.loglog(dt, np.exp(b) * dt ** a, label=f"line of best fit")
plt.xlabel('dt')
plt.ylabel('error')
plt.title('Forward Euler Error')
plt.legend()
plt.show()

# Store error values as a row vector named A2
A2 = np.array(error_vals)

# Store the slope of the line of best fit as a scalar named A3
A3 = a

# %%
# Question 1B
def huen(dydt, y0, dt):
    t = np.arange(0, t_last + dt, dt)
    y = np.zeros(len(t))
    y[0] = y0
    for i in range(1, len(t)):
        y[i] = y[i-1] + (dt / 2) * (dydt(y[i-1], t[i-1]) + dydt(y[i-1] + dydt(y[i-1], t[i-1]) * dt, t[i-1] + dt)) 
    return y

error_vals = []
A4 = []
for i in dt:
    t = np.arange(0, t_last + i, i)
    res = huen(dydt, y0, i)
    error = np.abs(res[-1] - ytrue(t[-1]))
    error_vals.append(error)
    if i == (1 / (2 ** 8)):
        A4 = res

#plot log of dt vs log of error with title "Huen's Method Error" and np.polyfit for line of best fit
plt.loglog(dt, error_vals, 'o')
a, b = np.polyfit(np.log(dt), np.log(error_vals), 1)
plt.loglog(dt, np.exp(b) * dt ** a, label=f"line of best fit")
plt.xlabel('dt')
plt.ylabel('error')
plt.title("Huen's Method Error")
plt.legend()
plt.show()

# Store error values as a row vector named A5
A5 = np.array(error_vals)

# Store the slope of the line of best fit as a scalar named A6
A6 = a

# %%
# Question 1C
def adams_predictor_corrector(dydt, y0, dt):
    t = np.arange(0, t_last + dt, dt)
    y = np.zeros(len(t))
    y[0] = y0
    y[1] = y[0] + dt * dydt(y[0] + dt/2 * dydt(y[0], t[0]), t[0] + dt/2)
    for i in range(2, len(t)):
        yp = y[i] + dt/2 * (3 * dydt(y[i], t[i]) - dydt(y[i-1], t[i-1]))
        y[i] = y[i] + dt/2 * (dydt(yp, t[i] + dt) + dydt(y[i], t[i]))
    return y

error_vals = []
A7 = []
for i in dt:
    t = np.arange(0, t_last + i, i)
    res = adams_predictor_corrector(dydt, y0, i)
    error = np.abs(res[-1] - ytrue(t[-1]))
    error_vals.append(error)
    if i == (1 / (2 ** 8)):
        A7 = res

# Store error values as a row vector named A8
A8 = np.array(error_vals)

# Store the slope of the line of best fit as a scalar named A9
A9 = a

# %%
# Presentation Mastery
# Plot log(E) vs log(dt) for all three methods on the same plot. Include lines with slope n where n is the order of the method.

#plot log of dt vs log of error with title "All Methods Error" and np.polyfit for line of best fit
plt.loglog(dt, A2, 'o', label="Forward Euler")
plt.loglog(dt, A5, 'o', label="Huen's Method")
plt.loglog(dt, A8, 'o', label="Adams Predictor Corrector")
a, b = np.polyfit(np.log(dt), np.log(A2), 1)
plt.loglog(dt, np.exp(b) * dt ** a, label=f"line of best fit for Forward Euler")
a, b = np.polyfit(np.log(dt), np.log(A5), 1)
plt.loglog(dt, np.exp(b) * dt ** a, label=f"line of best fit for Huen's Method")
a, b = np.polyfit(np.log(dt), np.log(A8), 1)
plt.loglog(dt, np.exp(b) * dt ** a, label=f"line of best fit for Adams Predictor Corrector")
plt.xlabel('dt')
plt.ylabel('error')
plt.title("All Methods Error")
plt.legend()
plt.show()

# %%
# Question 2 constants
epsilons = [0.1, 1, 20]
y_initial = np.sqrt(3)
dydt_initial = 1
t_last = 32
dt = 0.5

def vdp_derivatives(t, y):
    #let dydt = v and let y(t) = x
    x = y[0]
    v = y[1]
    return np.array([v, -1 * epsilon * (x*x - 1)*v - x])
# %%
# Question 2A
t = np.arange(0, t_last + dt, dt)
#Store the solution for each epsilon in a matrix with 3 columns called A10
A10 = np.zeros((len(t), 3))
for i in range(len(epsilons)):
    epsilon = epsilons[i]
    sol = scipy.integrate.solve_ivp(fun = vdp_derivatives, t_span=[t[0], t[-1]], y0 = [y_initial, dydt_initial], t_eval=t)
    A10[:, i] = sol.y[0, :]

# %%
# Question 2B

# %%
# Question 3

