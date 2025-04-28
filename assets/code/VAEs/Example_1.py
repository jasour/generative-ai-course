#https://github.com/rll/deepul/blob/master/homeworks/hw2/hw2.ipynb

# VAEs on 2D Data
# We will train a simple VAE on 2D data, and look at situations in which latents are being used or not being used (i.e. when posterior collapse occurs)

# Download and load up the starter code.
# if [ -d deepul ]; then rm -Rf deepul; fi
# git clone https://github.com/rll/deepul.git
# pip install ./deepul

# **********************************************************************************************
# **********************************************************************************************
# a) In this part, we train a VAE on data generated from a Gaussian with a full covariance matrix. 


# Construct and train a VAE with the following characteristics

# - 2D latent variables z with a standard normal prior, p(z) = N(0, I)
# - An approximate posterior qθ(z|x) = N(z; μθ(x), Σθ(x)I), where μθ(x) is the mean vector, 
#   and Σθ(x) is a diagonal covariance matrix
# - A decoder p(x|z) = N(x; μφ(z), Σφ(z)I), where μφ(z) is the mean vector, 
#   and Σφ(z) is a diagonal covariance matrix

# You will provide the following deliverables:

# 1. Over the course of training, record the average full negative ELBO, reconstruction loss 
#    E_x E_z~q(z|x)[-p(x|z)], and KL term E_x[D_KL(q(z|x) || p(z))] of the training data (per minibatch)
#    and test data (for your entire test set). Code is provided that automatically plots the training curves.

# 2. Report the final test set performance of your final model.

# 3. Samples of your trained VAE with (z ~ p(z), x ~ N(x; μφ(z), Σφ(z))) and without 
#    (z ~ p(z), x = μφ(z)) decoder noise.

""""
def q1(train_data, test_data, part, dset_id):
  
    train_data: An (n_train, 2) numpy array of floats
    test_data: An (n_test, 2) numpy array of floats

    (You probably won't need to use the two inputs below, but they are there
     if you want to use them)
    part: An identifying string ('a' or 'b') of which part is being run. Most likely
          used to set different hyperparameters for different datasets
    dset_id: An identifying number of which dataset is given (1 or 2). Most likely
               used to set different hyperparameters for different datasets

    Returns
    - a (# of training iterations, 3) numpy array of full negative ELBO, reconstruction loss E[-p(x|z)],
      and KL term E[KL(q(z|x) | p(z))] evaluated every minibatch
    - a (# of epochs + 1, 3) numpy array of full negative ELBO, reconstruciton loss E[-p(x|z)],
      and KL term E[KL(q(z|x) | p(z))] evaluated once at initialization and after each epoch
    - a numpy array of size (1000, 2) of 1000 samples WITH decoder noise, i.e. sample z ~ p(z), x ~ p(x|z)
    - a numpy array of size (1000, 2) of 1000 samples WITHOUT decoder noise, i.e. sample z ~ p(z), x = mu(z)
    

    Main CODE HERE 
    """

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
#from deepul.hw2_helper import visualize_q1_data, q1_save_results
from deepul.hw2_helper import *

# Visualize datasets
visualize_q1_data('a', 1)
visualize_q1_data('a', 2)

# Define VAE model
class VAE(nn.Module):
    def __init__(self, input_dim=2, latent_dim=2, hidden_dim=64):
        super(VAE, self).__init__()

        # Encoder: q(z|x) -> Outputs mean and log variance for reparameterization trick
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.mu_layer = nn.Linear(hidden_dim, latent_dim)
        self.log_var_layer = nn.Linear(hidden_dim, latent_dim)

        # Decoder: p(x|z) -> Outputs mean and log variance of reconstructed x
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.recon_mu = nn.Linear(hidden_dim, input_dim)
        self.recon_log_var = nn.Linear(hidden_dim, input_dim)  # Output log variance

    def encode(self, x):
        h = self.encoder(x)
        mu = self.mu_layer(h)
        log_var = self.log_var_layer(h)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.decoder(z)
        mu_x = self.recon_mu(h)
        log_var_x = self.recon_log_var(h)
        return mu_x, log_var_x

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        recon_mu, recon_log_var = self.decode(z)
        return recon_mu, recon_log_var, mu, log_var, z

# Define loss function for VAE
def loss_function(recon_mu, recon_log_var, x, mu, log_var):
    recon_loss = Normal(recon_mu, torch.exp(0.5 * recon_log_var)).log_prob(x).sum(dim=-1).mean()
    kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1).mean()
    return -recon_loss + kl_div, -recon_loss, kl_div

# Train the VAE
def q1(train_data, test_data, part, dset_id, epochs=50, batch_size=64, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

    model = VAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    training_logs = []
    epoch_logs = []

    for epoch in range(epochs + 1):
        model.train()
        batch_losses = []

        for x in train_loader:
            x = x.float().to(device)
            optimizer.zero_grad()

            recon_mu, recon_log_var, mu, log_var, _ = model(x)
            loss, recon_loss, kl_div = loss_function(recon_mu, recon_log_var, x, mu, log_var)

            loss.backward()
            optimizer.step()

            batch_losses.append([loss.item(), recon_loss.item(), kl_div.item()])

        batch_losses = np.array(batch_losses)
        training_logs.append(batch_losses.mean(axis=0))

        # Evaluate on test data
        model.eval()
        with torch.no_grad():
            test_losses = []

            for x in test_loader:
                x = x.float().to(device)
                recon_mu, recon_log_var, mu, log_var, _ = model(x)
                loss, recon_loss, kl_div = loss_function(recon_mu, recon_log_var, x, mu, log_var)
                test_losses.append([loss.item(), recon_loss.item(), kl_div.item()])

            test_losses = np.array(test_losses)
            epoch_logs.append(test_losses.mean(axis=0))

    # Sampling from trained VAE
    with torch.no_grad():
        z_samples = torch.randn(1000, 2).to(device)
        x_samples_with_noise_mu, x_samples_with_noise_logvar = model.decode(z_samples)
        x_samples_with_noise = Normal(x_samples_with_noise_mu, torch.exp(0.5 * x_samples_with_noise_logvar)).sample()

        x_samples_without_noise = x_samples_with_noise_mu

    return (
        np.array(training_logs),
        np.array(epoch_logs),
        x_samples_with_noise.cpu().numpy(),
        x_samples_without_noise.cpu().numpy()
    )

# Save results
q1_save_results('a', 1, q1)
q1_save_results('a', 2, q1)

# ***************************************************************************************************************
#****************************************************************************************************************
# b) In this part, we use your code from the previous part to train a VAE on data generated from a diagonal gaussian. 
# Execute the cell below to visualize the datasets (note that they may look the same, but notice the axes)

visualize_q1_data('b', 1)
visualize_q1_data('b', 2)

q1_save_results('b', 1, q1)
q1_save_results('b', 2, q1)