# %%
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
# %%
# Question 1 

# Define some constants
n = lambda x: x**2
L = 4
xp = [-L, L] # xspan, don't need to define stepsize
tol = 1e-6 # We want to find beta such that |y(x=1)| < tol

# Define ODE
def rhsfunc(x, y, epsilon):
    f1 = y[1]
    f2 = (n(x) - epsilon) * y[0]
    return np.array([f1, f2])

# Define our initial conditions
A = 1 # This is the shooting-method parameter that we will change
epsilon_start = 0 # This is our initial beta value, we will change it.

A1_through_A5 = []
A6 = []
# Make a loop over beta values to find more eigenvalue-eigenfunction pairs
for modes in range(5): # Try to find 5 modes
    epsilon = epsilon_start 
    depsilon = n(-L) / 100 # This is the amount we will decrease beta by each time we don't have an eigenvalue
                 # until we get an eigenvalue

    
    for j in range(1000):
        x_evals, step_size = np.linspace(-L, L, 20 * L + 1, retstep = True)
        initial_condition_one = np.sqrt(L**2 - epsilon_start)
        y0 = np.array([initial_condition_one, A])
        sol = scipy.integrate.solve_ivp(lambda x,y: rhsfunc(x, y, epsilon), xp, y0, t_eval = x_evals)
        y_sol = sol.y[0, :]
        y_prime_sol = sol.y[1, :]

        right_endpoint_derivative = -np.sqrt(L**2 - epsilon) * y_sol[-1]
        if np.abs(y_prime_sol[-1] - right_endpoint_derivative)<tol:
            print(f"We got the eigenvalue! epsilon = {epsilon}")
            # normalize the eigenfunction using the trapezoidal rule
            norm = scipy.integrate.trapz(y_sol**2, x_evals)
            y_sol = y_sol / np.sqrt(norm)
            A6.append(epsilon)
            break

        if (-1)**(modes)*(y_prime_sol[-1] - right_endpoint_derivative) > 0: 
            epsilon = epsilon + depsilon
        else:
            epsilon = epsilon - depsilon/2 
            depsilon = depsilon/2


    epsilon_start = epsilon + 0.1 # increase beta once we have found one mode.

    plt.plot(sol.t, y_sol, linewidth=2)
    plt.plot(sol.t, 0*sol.t, 'k')
    A1_through_A5.append(y_sol)
# %%
# Question 2
h = step_size
arr_size = 81
D = -1/(h**2) * np.ones(arr_size)
U = 2/(h**2) * np.ones(arr_size - 1) 
for i in range(U.size):
    U[i] = U[i] + (-L + ((i+1) * h))**2
L = -1/(h**2) * np.ones(arr_size - 2)
A = np.diag(D) + np.diag(U, k=1) + np.diag(L, k=2)

row_zero = np.zeros(arr_size)
row_zero[1] = 4/3
row_zero[2] = -1/3

row_N = np.zeros(arr_size)
row_N[-2] = 4/3
row_N[-3] = -1/3

#add row_zero to the top of A
A = np.vstack((row_zero, A))

#delete the last row of A
A = np.delete(A, -1, 0)

#replace the last row of A with row_N
A[-1] = row_N

#Get the eigenvalues and eigenvectors of A
eigenvalues, eigenvectors = np.linalg.eig(A)

#sort the eigenvalues and eigenvectors
idx = eigenvalues.argsort()[::1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:,idx]

#Save the first 5 eigenvectors to variables A7 through A11
A7 = eigenvectors[:,0]
A8 = eigenvectors[:,1]
A9 = eigenvectors[:,2]
A10 = eigenvectors[:,3]
A11 = eigenvectors[:,4]

#Save the first 5 eigenvalue as the variables A12
A12 = eigenvalues[0:5]

print(A7)

# %%
# Question 3
gamma = [0.05, -0.05]
tol = 1e-5
L = 3
xp = [-L, L]
x_evals, step_size = np.linspace(-L, L, 20 * L + 1, retstep = True)
A = 1
n = lambda x: x**2

#Define the ODE
def rhsfunc(x, y, epsilon, g):
    f1 = y[1]
    f2 = (gamma * y[0]**2 + n(x) - epsilon) * y[0]
    return np.array([f1, f2])

# for loop over two modes
#    for loop for shooting
#       update initial condition
#       solve ODE
#       compute norm and boundary condition
#       if norm and boundary condition met
#          break
#       else
#          A = A/sqrt(norm)
      
#       update initial condition with new A
#       solve ODE
#       compute norm and boundary condition
#       if norm and boundary condition met
#          break
#       else
#          change epsilon accordingly

#    epsilon_start = epsilon + 0.1
#    A remains the same

for gam in gamma:
    epsilon_start = 0
    for modes in range(2):
        epsilon = epsilon_start
        depsilon = n(-L) / 100
        for j in range(1000):
            initial_condition_one = np.sqrt(L**2 - epsilon) * A
            y0 = np.array([initial_condition_one, A])
            sol = scipy.integrate.solve_ivp(lambda x,y: rhsfunc(x, y, epsilon, gam), xp, y0, t_eval = x_evals)
            y_sol = sol.y[0, :]
            y_prime_sol = sol.y[1, :]
            right_endpoint_derivative = -np.sqrt(L**2 - epsilon) * y_sol[-1]
            norm = scipy.integrate.trapz(y_sol**2, x_evals)
            if np.abs(y_prime_sol[-1] - right_endpoint_derivative)<tol and np.abs(norm - 1) < tol:
                print(f"We got the eigenvalue! epsilon = {epsilon}")
                break
            else:
                A = A / np.sqrt(norm)
            
            y0 = np.array([initial_condition_one, A])
            sol = scipy.integrate.solve_ivp(lambda x,y: rhsfunc(x, y, epsilon, gam), xp, y0, t_eval = x_evals)
            y_sol = sol.y[0, :]
            y_prime_sol = sol.y[1, :]
            right_endpoint_derivative = -np.sqrt(L**2 - epsilon) * y_sol[-1]
            norm = scipy.integrate.trapz(y_sol**2, x_evals)
            if np.abs(y_prime_sol[-1] - right_endpoint_derivative)<tol and np.abs(norm - 1) < tol:
                print(f"We got the eigenvalue! epsilon = {epsilon}")
                break
            else:
                if (-1)**(modes)*(y_prime_sol[-1] - right_endpoint_derivative) > 0: 
                    epsilon = epsilon + depsilon
                else:
                    epsilon = epsilon - depsilon/2 
                    depsilon = depsilon/2
            
        epsilon_start = epsilon + 0.1 # increase beta once we have found one mode.
        plt.plot(sol.t, y_sol, linewidth=2)
        plt.plot(sol.t, 0*sol.t, 'k')

# %%
