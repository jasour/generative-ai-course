# https://diffusion.csail.mit.edu/
# “Generative AI With Stochastic Differential Equations” MIT, 6.S184/6.S975, IAP 2025.
# Langevin Dynamics 2

import torch, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from abc import ABC, abstractmethod
from tqdm import tqdm
from matplotlib.axes import Axes
import torch.distributions as D
from torch.func import vmap, jacrev

device = torch.device("cpu")

class SDE(ABC):
    @abstractmethod
    def drift_coefficient(self, xt, t): pass
    @abstractmethod
    def diffusion_coefficient(self, xt, t): pass

class Simulator(ABC):
    @abstractmethod
    def step(self, xt, t, dt): pass
    @torch.no_grad()
    def simulate(self, x, ts):
        for i in range(len(ts)-1):
            x = self.step(x, ts[i], ts[i+1]-ts[i])
        return x
    @torch.no_grad()
    def simulate_with_trajectory(self, x, ts):
        xs = [x.clone()]
        for i in tqdm(range(len(ts)-1)):
            x = self.step(x, ts[i], ts[i+1]-ts[i])
            xs.append(x.clone())
        return torch.stack(xs, dim=1)

class EulerMaruyamaSimulator(Simulator):
    def __init__(self, sde): self.sde = sde
    def step(self, xt, t, h):
        return xt + self.sde.drift_coefficient(xt, t)*h + self.sde.diffusion_coefficient(xt, t)*torch.sqrt(h)*torch.randn_like(xt)

class Density(ABC):
    @abstractmethod
    def log_density(self, x): pass
    def score(self, x):
        x = x.unsqueeze(1)
        return vmap(jacrev(self.log_density))(x).squeeze((1,2,3))

class Sampleable(ABC):
    @abstractmethod
    def sample(self, num_samples): pass

def imshow_density(density, bins, scale, ax=None, **kwargs):
    ax = ax or plt.gca()
    x = torch.linspace(-scale, scale, bins).to(device)
    y = torch.linspace(-scale, scale, bins).to(device)
    X, Y = torch.meshgrid(x, y)
    xy = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=-1)
    ax.imshow(density.log_density(xy).reshape(bins, bins).T.cpu(),
              extent=[-scale, scale, -scale, scale], origin='lower', **kwargs)

class Gaussian(torch.nn.Module, Sampleable, Density):
    def __init__(self, mean, cov):
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("cov", cov)
    @property
    def distribution(self): return D.MultivariateNormal(self.mean, self.cov, validate_args=False)
    def sample(self, num_samples): return self.distribution.sample((num_samples,))
    def log_density(self, x): return self.distribution.log_prob(x).view(-1,1)

class GaussianMixture(torch.nn.Module, Sampleable, Density):
    def __init__(self, means, covs, weights):
        super().__init__()
        self.register_buffer("means", means)
        self.register_buffer("covs", covs)
        self.register_buffer("weights", weights)
    @property
    def distribution(self):
        return D.MixtureSameFamily(D.Categorical(probs=self.weights, validate_args=False),
                                   D.MultivariateNormal(loc=self.means, covariance_matrix=self.covs, validate_args=False),
                                   validate_args=False)
    def log_density(self, x): return self.distribution.log_prob(x).view(-1,1)
    def sample(self, num_samples): return self.distribution.sample(torch.Size((num_samples,)))
    @classmethod
    def random_2D(cls, nmodes, std, scale=10.0, seed=0.0):
        torch.manual_seed(seed)
        means = (torch.rand(nmodes,2)-0.5)*scale
        covs = torch.diag_embed(torch.ones(nmodes,2))*std**2
        weights = torch.ones(nmodes)
        return cls(means, covs, weights)
    @classmethod
    def symmetric_2D(cls, nmodes, std, scale=10.0):
        angles = torch.linspace(0,2*np.pi, nmodes+1)[:nmodes]
        means = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)*scale
        covs = torch.diag_embed(torch.ones(nmodes,2)*std**2)
        weights = torch.ones(nmodes)/nmodes
        return cls(means, covs, weights)

densities = {
    "Gaussian": Gaussian(torch.zeros(2), 10*torch.eye(2)).to(device),
    "Random Mixture": GaussianMixture.random_2D(5,1.0,20.0,seed=3.0).to(device),
    "Symmetric Mixture": GaussianMixture.symmetric_2D(5,1.0,8.0).to(device)
}

class LangevinSDE(SDE):
    def __init__(self, sigma, density):
        self.sigma = sigma; self.density = density
    def drift_coefficient(self, xt, t):
        return 0.5*self.sigma**2*self.density.score(xt)
    def diffusion_coefficient(self, xt, t):
        return self.sigma*torch.ones_like(xt)

def every_nth_index(n_timesteps, n):
    return torch.cat([torch.arange(0, n_timesteps-1, n), torch.tensor([n_timesteps-1])]) if n>1 else torch.arange(n_timesteps)

def graph_dynamics(n_samples, source_distribution, simulator, density, ts, plot_every, bins, scale):
    x0 = source_distribution.sample(n_samples)
    xts = simulator.simulate_with_trajectory(x0, ts)
    indices = every_nth_index(len(ts), plot_every)
    plot_ts = ts[indices]
    plot_xts = xts[:,indices]
    fig, axes = plt.subplots(2, len(plot_ts), figsize=(8*len(plot_ts),16))
    axes = axes.reshape(2, len(plot_ts))
    for i in range(len(plot_ts)):
        t = plot_ts[i].item()
        xt = plot_xts[:,i]
        ax_scat = axes[0,i]
        imshow_density(density, bins, scale, ax_scat, vmin=-15, alpha=0.25, cmap=plt.get_cmap('Blues'))
        ax_scat.scatter(xt[:,0].cpu().detach().numpy(), xt[:,1].cpu().detach().numpy(),
                        marker='x', color='black', alpha=0.75, s=15)
        ax_scat.set_title(f'Samples at t={t:.1f}', fontsize=15); ax_scat.set_xticks([]); ax_scat.set_yticks([])
        ax_kde = axes[1,i]
        imshow_density(density, bins, scale, ax_kde, vmin=-15, alpha=0.5, cmap=plt.get_cmap('Blues'))
        sns.kdeplot(x=xt[:,0].cpu().detach().numpy(), y=xt[:,1].cpu().detach().numpy(),
                    alpha=0.5, ax=ax_kde, color='grey')
        ax_kde.set_title(f'Density at t={t:.1f}', fontsize=15)
        ax_kde.set_xticks([]); ax_kde.set_yticks([]); ax_kde.set_xlabel(""); ax_kde.set_ylabel("")
    plt.show()

target = GaussianMixture.random_2D(5,0.75,15.0,seed=3.0).to(device)
sde = LangevinSDE(0.6, target)
simulator = EulerMaruyamaSimulator(sde)
graph_dynamics(1000, Gaussian(torch.zeros(2), 20*torch.eye(2)).to(device),
               simulator, target, torch.linspace(0,5.0,1000).to(device), 334, 200, 15)