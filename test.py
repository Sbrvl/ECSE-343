import numpy as np
from circuit_simulator import CircuitSimulator
from helper_functions import plot_data, create_dataset, save_dataset

# ==========================================
# 1. Simulation Parameters & Constants
# ==========================================
R_test = 2.5e3          # Test resistance in Ohms
C_test = 3e-6           # Test capacitance in Farads
amplitude = 5           # Source voltage amplitude (V)
frequency = 60          # Source frequency (Hz)
delta_t = 1e-4          # Time step size (s)
T = 0.05                # Total simulation duration (s)

# ==========================================
# 2. Data Loading & Initialization
# ==========================================
# x_init: Initial state vector  [V1, V2, V_3, I_E]
x_init = np.zeros((4,)) 

# Load true measurements 
x_true = np.loadtxt('measurements.csv', delimiter=',')

# Dataset generation settings
num_samples = 2000       # Integer
noisy_dataset = True     # Boolean

# ==========================================
# 3. Forward Simulation (Backward Euler)
# ==========================================
# Initialize the Modified Nodal Analysis (MNA) object
mna = CircuitSimulator(amplitude, frequency, R_test, C_test)

# Solve the circuit using the Backward Euler numerical integration method
x_test, tpoints = mna.BEuler(x_init, delta_t, T, noise = True)

# Visualize the simulation results
plot_data(x_test, tpoints)

# ==========================================
# 4. Parameter Estimation (Gauss-Newton)
# ==========================================
R_guess = 2.5e3         # Initial guess for Resistor
C_guess = 3e-6          # Initial guess for Capacitor

# Initialize the Modified Nodal Analysis (MNA) object
mna = CircuitSimulator(amplitude, frequency, R_guess, C_guess)

# Run Gauss-Newton optimization 
R_pred, C_pred, cost = mna.GaussNewton(R_guess, C_guess, x_init, x_true, delta_t, T, max_iter=10)

print(f"Predicted resistor value:  {R_pred}.")
print(f"Predicted capacitor value: {C_pred}.")
print(f"Cost                     : {cost}.")

# ==========================================
# 5. Dataset Generation for Machine Learning
# ==========================================
# Generate a batch of simulations with varying parameters
X, y = create_dataset(num_samples, amplitude, frequency, delta_t, T, noise=noisy_dataset)

# Export data for training 
save_dataset(X, y, log_targets=True)
