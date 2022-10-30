# %%
import numpy as np
import matplotlib.pyplot as plt

#%%
psi = 0
mass = 1
PlanckConstant = 6.62607004e-34
v = lambda k, x: (k * x ** 2) / 2
L = 4
xspan = np.linspace(-4, 4, 20 * L  + 11)
