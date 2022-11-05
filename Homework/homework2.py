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
        initial_condition_one = np.sqrt(L**2 - epsilon)
        y0 = np.array([1, initial_condition_one])
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
            A1_through_A5.append(np.abs(y_sol))
            break

        if (-1)**(modes)*(y_prime_sol[-1] - right_endpoint_derivative) > 0: 
            epsilon = epsilon + depsilon
        else:
            epsilon = epsilon - depsilon/2 
            depsilon = depsilon/2


    epsilon_start = epsilon + 0.1 # increase beta once we have found one mode.

    plt.plot(sol.t, np.abs(y_sol), linewidth=2)
    plt.plot(sol.t, 0*sol.t, 'k')

#Save the absolute value of the eigenvalues to variables A1 through A5 as column vectors
A1 = A1_through_A5[0]
A2 = A1_through_A5[1]
A3 = A1_through_A5[2]
A4 = A1_through_A5[3]
A5 = A1_through_A5[4]

#Cast A1 through A5 into column vectors
A1 = A1.reshape(-1, 1)
A2 = A2.reshape(-1, 1)
A3 = A3.reshape(-1, 1)
A4 = A4.reshape(-1, 1)
A5 = A5.reshape(-1, 1)

#Cast A6 into a row vector
A6 = np.array(A6)
A6 = A6.reshape(1, -1)

# %%
# Question 2
h = step_size
arr_size = 79
D = 2/(h**2) * np.ones(arr_size) 
for i in range(D.size):
    D[i] = D[i] + (-L + ((i+1) * h))**2
U = -1/(h**2) * np.ones(arr_size - 1)
L = -1/(h**2) * np.ones(arr_size - 1)
A = np.diag(D) + np.diag(U, k=1) + np.diag(L, k=-1)

print(A)

#Get the eigenvalues and eigenvectors of A
eigenvalues, eigenvectors = np.linalg.eig(A)
eigenvectors_final = np.zeros((81, 79))
#loop over all the eigenvectors:
for i in range(eigenvectors.shape[1]):
    new_eigenvector = []
    #get the 0th element as the (4 * first element of the eignvector - second element) / 3
    y0 = (4 * eigenvectors[0, i] - eigenvectors[1, i]) / 3
    #get the last element as the (4 * the last element of the eignvector - second to last element) / 3
    y1 = (4 * eigenvectors[-1, i] - eigenvectors[-2, i]) / 3
    #add y0 to the beginning of the eigenvector
    new_eigenvector.append(y0)
    #add the eigenvector to the new eigenvector
    new_eigenvector.extend(eigenvectors[:, i])
    #add y1 to the end of the eigenvector
    new_eigenvector = np.append(new_eigenvector, y1)
    #normalize the eigenvector
    norm = scipy.integrate.trapz(np.power(new_eigenvector, 2), x_evals)
    new_eigenvector = new_eigenvector / np.sqrt(norm)
    #add the eigenvector to the list of eigenvectors as a column vector
    eigenvectors_final[:, i] = new_eigenvector

eigenvectors = np.array(eigenvectors_final)
#sort the eigenvalues and eigenvectors
idx = eigenvalues.argsort()[::1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:,idx]

for i in range(5):
    plt.plot(np.arange(-4, 4 + h, h), eigenvectors[:, i], label = f"eigenvalue = {eigenvalues[i]}")
    plt.plot(sol.t, 0*sol.t, 'k')
    plt.legend()


#Save the first 5 abolute values of the eigenvectors to variables A7 through A11
A7 = np.abs(eigenvectors[:,0])
A8 = np.abs(eigenvectors[:,1])
A9 = np.abs(eigenvectors[:,2])
A10 = np.abs(eigenvectors[:,3])
A11 = np.abs(eigenvectors[:,4])

# cast A7 through A11 as column vectors
A7 = A7.reshape(-1, 1)
A8 = A8.reshape(-1, 1)
A9 = A9.reshape(-1, 1)
A10 = A10.reshape(-1, 1)
A11 = A11.reshape(-1, 1)

#Save the first 5 eigenvalue as the variables A12
A12 = eigenvalues[0:5]

# Case A12 as a row vector
A12 = A12.reshape(1, -1)

print(A12)
# %%
# Question 3
gamma = [0.05, -0.05]
tol = 1e-5
L = 3
xp = [-L, L]
x_evals, step_size = np.linspace(-L, L, 20 * L + 1, retstep = True)
A = 0.001
n = lambda x: x**2

#Define the ODE
def rhsfunc(x, y, epsilon, gamma):
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

A13_A14_A16_A17 = []
A15_A18 = []
for gam in gamma:
    epsilon_start = 0
    for modes in range(2):
        epsilon = epsilon_start
        depsilon = n(-L) / 100
        for j in range(1000):
            initial_condition_one = np.sqrt(L**2 - epsilon) * A
            y0 = np.array([A, initial_condition_one])
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
            
            y0 = np.array([A, initial_condition_one])
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
        A13_A14_A16_A17.append(y_sol)
        A15_A18.append(epsilon)

# Save the eigenfunctions as A13, A14, A16, A17
A13 = np.abs(A13_A14_A16_A17[0])
A14 = np.abs(A13_A14_A16_A17[1])
A16 = np.abs(A13_A14_A16_A17[2])
A17 = np.abs(A13_A14_A16_A17[3])

# Save the eigenvalues as A15, A18
A15 = np.array((A15_A18[0], A15_A18[1]))
A18 = np.array((A15_A18[2], A15_A18[3]))

# Cast A13 through A17 as column vectors
A13 = A13.reshape(-1, 1)
A14 = A14.reshape(-1, 1)
A16 = A16.reshape(-1, 1)
A17 = A17.reshape(-1, 1)

# Cast A15 and A18 as row vectors
A15 = A15.reshape(1, -1)
A18 = A18.reshape(1, -1)


# %%
