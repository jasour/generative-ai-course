# https://diffusion.csail.mit.edu/
# “Generative AI With Stochastic Differential Equations” MIT, 6.S184/6.S975, IAP 2025.
# Brownian Motion

import torch
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import Optional
from tqdm import tqdm

# Select device (MPS for Mac, CPU otherwise)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ******** ODE and SDE class: abstract base class (ABC)  ********
class ODE(ABC):
    @abstractmethod
    def drift_coefficient(self, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor: pass

class SDE(ABC):
    @abstractmethod
    def drift_coefficient(self, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor: pass
    @abstractmethod
    def diffusion_coefficient(self, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor: pass

# ******** Simulator abstract base class (ABC)  ********
class Simulator(ABC):
    @abstractmethod
    def step(self, xt: torch.Tensor, t: torch.Tensor, dt: torch.Tensor): pass

    @torch.no_grad()
    def simulate(self, x: torch.Tensor, ts: torch.Tensor):
        for t_idx in range(len(ts) - 1):
            x = self.step(x, ts[t_idx], ts[t_idx + 1] - ts[t_idx])
        return x

    @torch.no_grad()
    def simulate_with_trajectory(self, x: torch.Tensor, ts: torch.Tensor):
        xs = [x.clone()]
        for t_idx in tqdm(range(len(ts) - 1)):
            x = self.step(x, ts[t_idx], ts[t_idx + 1] - ts[t_idx])
            xs.append(x.clone())
        return torch.stack(xs, dim=1)

# ******** Euler Simulators : uses ODE and SDE instances ********
class EulerSimulator(Simulator):
    def __init__(self, ode: ODE): self.ode = ode
    def step(self, xt, t, h): return xt + self.ode.drift_coefficient(xt, t) * h

class EulerMaruyamaSimulator(Simulator):
    def __init__(self, sde: SDE): self.sde = sde
    def step(self, xt, t, h):
        return xt + self.sde.drift_coefficient(xt, t) * h + \
               self.sde.diffusion_coefficient(xt, t) * torch.sqrt(h) * torch.randn_like(xt)

# ******** Brownian Motion ********
class BrownianMotion(SDE):
    def __init__(self, sigma: float): self.sigma = sigma #  diffusion coefficient
    def drift_coefficient(self, xt, t): return torch.zeros_like(xt)
    def diffusion_coefficient(self, xt, t): return self.sigma * torch.ones_like(xt)

# ******** Visualization ********
def plot_trajectories_1d(x0, simulator, ts, ax=None):
    if ax is None: ax = plt.gca()
    for traj in simulator.simulate_with_trajectory(x0, ts):
        ax.plot(ts.cpu(), traj[:, 0].cpu())

# ******** Main Execution ********
sigma, x0 = 1.0, torch.zeros(5, 1).to(device)
brownian_motion = BrownianMotion(sigma)
simulator = EulerMaruyamaSimulator(brownian_motion)
ts = torch.linspace(0, 5, 500).to(device) # creates 500 evenly spaced time points between 0.0 and 5.0, h=0.01

plt.figure(figsize=(8, 8))
plt.title(r'Trajectories of Brownian Motion with $\sigma=$' + str(sigma), fontsize=18)
plt.xlabel(r'Time ($t$)', fontsize=18)
plt.ylabel(r'$X_t$', fontsize=18)
plot_trajectories_1d(x0, simulator, ts)
plt.show()