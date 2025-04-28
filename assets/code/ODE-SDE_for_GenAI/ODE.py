import numpy as np
import matplotlib.pyplot as plt

# Define the ODE: dx/dt = -alpha * x
def ode_drift(x, t, alpha=0.5):
    return -alpha * x

# Euler method for ODE
def euler_ode(x0, t, alpha):
    dt = t[1] - t[0]
    x = np.zeros_like(t)
    x[0] = x0
    for i in range(1, len(t)):
        x[i] = x[i-1] + dt * ode_drift(x[i-1], t[i-1], alpha)
    return x

# Simulation parameters
T = 5  # Total time
dt = 0.01  # Time step
t = np.arange(0, T, dt)
x0 = 1  # Initial condition

# Solve ODE
x = euler_ode(x0, t, alpha=0.5)

# Plot results
plt.figure(figsize=(8, 4))
plt.plot(t, x, label="ODE Solution (Euler)")
plt.xlabel("Time")
plt.ylabel("x(t)")
plt.title("ODE Solution using Euler's Method")
plt.legend()
plt.show()