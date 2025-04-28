# https://github.com/rll/deepul/tree/master/homeworks/hw4

# Download and load up the starter code.
# if [ -d deepul ]; then rm -Rf deepul; fi
# git clone https://github.com/rll/deepul.git
# pip install ./deepul
# pip install scikit-learn


from deepul.hw4_helper import *
import warnings
warnings.filterwarnings('ignore')

#visualize the datasets
visualize_q1_dataset()

#***************************************************************************************************
# we will train a continuous-time variant of the diffusion prompt. 
# In practice training objectives and code between discrete-time and continuous-time diffusion models are similar.
#
#  Given a data element x and neural net fθ(x, t), we implement the following diffusion training steps:
# 1. Sample the diffusion timestep: t ~ Uniform(0, 1)
# 2. Compute the noise-strength following a cosine schedule: 
#    α_t = cos(π/2 * t), σ_t = sin(π/2 * t)
# 3. Apply the forward process - Sample noise ϵ ~ N(0, I) (same shape as x) and compute noised x_t:
#    x_t = α_t * x + σ_t * ϵ
# 4. Estimate ϵ^ = fθ(x_t, t)
# 5. Optimize the loss L = ||ϵ - ϵ^||²₂. 
#    Here, it suffices to just take the mean over all dimensions.

# Note that for the case of continuous-time diffusion, 
# the forward process is x_0 → x_1 and the reverse process is x_1 → x_0

# Use an MLP for fθ to optimize the loss. You may find the following details helpful:

# - Normalize the data using mean and std computed from the train dataset
# - Train 100 epochs, batch size 1024, Adam with LR 1e-3 (100 warmup steps, cosine decay to 0)
# - MLP with 4 hidden layers and hidden size 64
# - Condition on t by concatenating it with input x 
#   (i.e., 2D x + 1D t = 3D cat(x, t))

# To sample, implement the standard DDPM sampler. 
# You may find the equation from the DDIM paper helpful, rewritten and re-formatted here for convenience.

# x_t-1 = α_t-1 * ((x_t - σ_t * ϵ^) / α_t) + sqrt(σ²_t-1 - η²_t * α²_t-1) * ϵ^ + η_t * ϵ_t

# where ϵ_t ~ N(0, I) is random Gaussian noise. 
# For DDPM, let η_t = σ_t-1 / σ_t * sqrt(1 - α²_t / α²_t-1)


# Step 1: Import necessary libraries
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# Step 2: Define the MLP network for fθ(x, t)
class DiffusionMLP(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, output_dim=2):
        super(DiffusionMLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)  # Output same shape as input x
        )

    def forward(self, x, t):
        t = t.unsqueeze(1)  # Ensure t has correct shape (batch_size, 1)
        xt = torch.cat([x, t], dim=1)  # Concatenate x and t
        return self.model(xt)

# Step 3: Define the diffusion training function
def train_diffusion_model(train_data, test_data, num_epochs=100, batch_size=1024, lr=1e-3):
    # Convert data to PyTorch tensors
    train_data = torch.tensor(train_data, dtype=torch.float32)
    test_data = torch.tensor(test_data, dtype=torch.float32)

    # Normalize using training data statistics
    mean, std = train_data.mean(0), train_data.std(0)
    train_data = (train_data - mean) / std
    test_data = (test_data - mean) / std

    # Create DataLoaders
    train_loader = DataLoader(TensorDataset(train_data), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(test_data), batch_size=batch_size, shuffle=False)

    # Initialize model, optimizer, and loss function
    model = DiffusionMLP().to("mps" if torch.cuda.is_available() else "cpu")
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # Learning rate scheduler with cosine decay
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Track losses
    train_losses, test_losses = [], []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0

        for (x,) in train_loader:
            x = x.to("mps" if torch.cuda.is_available() else "cpu")
            
            # Sample t ~ Uniform(0,1)
            t = torch.rand((x.shape[0],), device=x.device)

            # Compute noise strength α_t, σ_t
            alpha_t = torch.cos(0.5 * np.pi * t).unsqueeze(1)
            sigma_t = torch.sin(0.5 * np.pi * t).unsqueeze(1)

            # Sample Gaussian noise ε ~ N(0, I)
            epsilon = torch.randn_like(x)

            # Compute noised x_t
            x_t = alpha_t * x + sigma_t * epsilon

            # Predict noise
            epsilon_hat = model(x_t, t)

            # Compute loss ||ε - ε^||²
            loss = loss_fn(epsilon_hat, epsilon)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            epoch_loss += loss.item()

        scheduler.step()

        # Compute test loss
        model.eval()
        with torch.no_grad():
            test_loss = 0
            for (x,) in test_loader:
                x = x.to("mps" if torch.cuda.is_available() else "cpu")
                t = torch.rand((x.shape[0],), device=x.device)
                alpha_t = torch.cos(0.5 * np.pi * t).unsqueeze(1)
                sigma_t = torch.sin(0.5 * np.pi * t).unsqueeze(1)
                epsilon = torch.randn_like(x)
                x_t = alpha_t * x + sigma_t * epsilon
                epsilon_hat = model(x_t, t)
                test_loss += loss_fn(epsilon_hat, epsilon).item()

        test_losses.append(test_loss / len(test_loader))

        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {epoch_loss/len(train_loader):.6f}, Test Loss: {test_losses[-1]:.6f}")

    return train_losses, test_losses, model

# Step 4: Implement DDPM Sampling Process

def ddpm_sampling(model, num_samples=2000, num_steps_list=None):
    if num_steps_list is None:
        num_steps_list = np.power(2, np.linspace(0, 9, 9)).astype(int)

    device = "mps" if torch.cuda.is_available() else "cpu"
    model.to(device)

    all_samples = []
    all_timesteps = []  # Store timesteps

    for num_steps in num_steps_list:
        ts = torch.linspace(1 - 1e-4, 1e-4, num_steps + 1).to(device)
        x = torch.randn((num_samples, 2), device=device)  # Start from Gaussian noise

        sample_timesteps = np.zeros(num_samples)  # Store the final timesteps

        for i in range(num_steps):
            t = ts[i].repeat(num_samples, 1)
            tm1 = ts[i + 1].repeat(num_samples, 1)

            epsilon_hat = model(x, t.squeeze(1))

            alpha_t = torch.cos(0.5 * np.pi * t)
            sigma_t = torch.sin(0.5 * np.pi * t)
            alpha_tm1 = torch.cos(0.5 * np.pi * tm1)
            sigma_tm1 = torch.sin(0.5 * np.pi * tm1)

            eta_t = (sigma_tm1 / sigma_t) * torch.sqrt(1 - alpha_t**2 / alpha_tm1**2)
            noise = torch.randn_like(x)

            # Compute x_{t-1} using the DDPM update rule
            x = alpha_tm1 * ((x - sigma_t * epsilon_hat) / alpha_t) + \
                torch.sqrt(torch.clamp(sigma_tm1**2 - eta_t**2 * alpha_tm1**2, min=0)) * epsilon_hat + \
                eta_t * noise

            sample_timesteps = t.cpu().numpy().flatten()  # Store last used timestep

        all_samples.append(x.detach().cpu().numpy())
        all_timesteps.append(sample_timesteps)  # Store corresponding timesteps

    return np.array(all_samples), np.array(all_timesteps)

def visualize_generated_samples(samples, timesteps, num_steps_list):
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes = axes.flatten()
    
    for i, (sample_set, timestep_set, num_steps) in enumerate(zip(samples, timesteps, num_steps_list)):
        ax = axes[i]
        scatter = ax.scatter(sample_set[:, 0], sample_set[:, 1], c=timestep_set, cmap='viridis', alpha=0.7)
        ax.set_title(f"Samples with {num_steps} Steps")
        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")

        # Add colorbar to show timestep values
        #cbar = plt.colorbar(scatter, ax=ax)
        #cbar.set_label("Time Step (t)")
    
    plt.tight_layout()
    plt.show()


# Step 5: Wrap everything inside q1 function
def q1(train_data, test_data):
    """
    train_data: A (100000, 2) numpy array of 2D points
    test_data: A (10000, 2) numpy array of 2D points

    Returns
    - a (# of training iterations,) numpy array of train losses evaluated every minibatch
    - a (# of num_epochs + 1,) numpy array of test losses evaluated at the start of training and the end of every epoch
    - a numpy array of size (9, 2000, 2) of samples drawn from your model.
      Draw 2000 samples for each of 9 different number of diffusion sampling steps
    """
    train_losses, test_losses, model = train_diffusion_model(train_data, test_data)
    num_steps_list = np.power(2, np.linspace(0, 9, 9)).astype(int)
    all_samples, all_timesteps = ddpm_sampling(model, num_samples=2000, num_steps_list=num_steps_list)

    visualize_generated_samples(all_samples, all_timesteps, num_steps_list)

    return np.array(train_losses), np.array(test_losses), all_samples

# Execute and save results
q1_save_results(q1)