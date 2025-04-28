#https://github.com/rll/deepul/blob/master/homeworks/hw2/hw2.ipynb

# VAEs on Images
 

# Download and load up the starter code.
# if [ -d deepul ]; then rm -Rf deepul; fi
# git clone https://github.com/rll/deepul.git
# pip install ./deepul

# **********************************************************************************************
# **********************************************************************************************

# We will train different VAE models on image datasets (CIFAR-10) 

# In this part, we implement a standard VAE with the following characteristics:

# - 16-dim latent variables z with standard normal prior p(z) = N(0, I)
# - An approximate posterior qθ(z|x) = N(z; μθ(x), Σθ(x)), where μθ(x) is the mean vector, 
#   and Σθ(x) is a diagonal covariance matrix
# - A decoder p(x|z) = N(x; μφ(z), I), where μφ(z) is the mean vector. 
#   (We are not learning the covariance of the decoder)

# You can play around with different architectures and try for better results, 
# but the following encoder/decoder architecture below suffices 
# (Note that image input is always 32 × 32).

# Convolutional and fully connected layers used in the architecture:
# conv2d(in_channels, out_channels, kernel_size, stride, padding)
# transpose_conv2d(in_channels, out_channels, kernel_size, stride, padding)
# linear(in_dim, out_dim)

# Encoder
# conv2d(3, 32, 3, 1, 1)
# relu()
# conv2d(32, 64, 3, 2, 1)  # Output: 16 × 16
# relu()
# conv2d(64, 128, 3, 2, 1)  # Output: 8 × 8
# relu()
# conv2d(128, 256, 3, 2, 1)  # Output: 4 × 4
# relu()
# flatten()  # Output: 16
# linear(4 * 4 * 256, 2 * latent_dim)

# Decoder
# linear(latent_dim, 4 * 4 * 128)
# relu()
# reshape(4, 4, 128)
# transpose_conv2d(128, 128, 4, 2, 1)  # Output: 8 × 8
# relu()
# transpose_conv2d(128, 64, 4, 2, 1)  # Output: 16 × 16
# relu()
# transpose_conv2d(64, 32, 4, 2, 1)  # Output: 32 × 32
# relu()
# conv2d(32, 3, 3, 1, 1)

"""
-When computing reconstruction loss and KL loss, average over the batch dimension and sum over the feature dimension
-When computing reconstruction loss, it suffices to just compute MSE between the reconstructed x and true x 
- Use batch size 128, learning rate 10^(-3), and an Adam optimizer

Deliverables: 
- Over the course of training, record the average full negative ELBO, reconstruction loss, and KL term of the training data (per minibatch) and test data (for your entire test set). Code is provided that automatically plots the training curves.
- Report the final test set performance of your final model
- 100 samples from your trained VAE
- 50 real-image / reconstruction pairs (for some x, encode and then decode)
- 10 interpolations of 10 images from your trained VAE (100 images total)


def q2_a(train_data, test_data, dset_id):
    
    train_data: An (n_train, 32, 32, 3) uint8 numpy array of color images with values in {0, ..., 255}
    test_data: An (n_test, 32, 32, 3) uint8 numpy array of color images with values in {0, ..., 255}
    dset_id: An identifying number of which dataset is given (1 or 2). Most likely
               used to set different hyperparameters for different datasets

    Returns
    - a (# of training iterations, 3) numpy array of full negative ELBO, reconstruction loss E[-p(x|z)],
      and KL term E[KL(q(z|x) | p(z))] evaluated every minibatch
    - a (# of epochs + 1, 3) numpy array of full negative ELBO, reconstruciton loss E[-p(x|z)],
      and KL term E[KL(q(z|x) | p(z))] evaluated once at initialization and after each epoch
    - a (100, 32, 32, 3) numpy array of 100 samples from your VAE with values in {0, ..., 255}
    - a (100, 32, 32, 3) numpy array of 50 real image / reconstruction pairs
      FROM THE TEST SET with values in {0, ..., 255}
    - a (100, 32, 32, 3) numpy array of 10 interpolations of length 10 between
      pairs of test images. The output should be those 100 images flattened into
      the specified shape with values in {0, ..., 255}
    
     Main CODE

q2_save_results('a', 1, q2_a)

"""


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
from deepul.hw2_helper import *

# Visualize CIFAR-10 dataset
visualize_cifar10()

# Define the VAE model with CNN Encoder-Decoder
class VAE(nn.Module):
    def __init__(self, latent_dim=16):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim

        # Encoder: q(z|x)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),  # 32x32 -> 32x32
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 32x32 -> 16x16
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 16x16 -> 8x8
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # 8x8 -> 4x4
            nn.ReLU(),
            nn.Flatten()  # Output shape: [B, 256*4*4 = 4096]
        )

        # Update fully connected layers to accept 4096 features
        self.fc_mu = nn.Linear(4096, latent_dim)
        self.fc_logvar = nn.Linear(4096, latent_dim)

        # Decoder: p(x|z)
        self.fc_decode = nn.Linear(latent_dim, 4 * 4 * 128)
        self.decoder = nn.Sequential(
            nn.ReLU(),
            nn.Unflatten(1, (128, 4, 4)),  # Reshape to (B, 128, 4, 4)
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),  # -> (B, 128, 8, 8)
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   # -> (B, 64, 16, 16)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),    # -> (B, 32, 32, 32)
            nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1),              # -> (B, 3, 32, 32)
            nn.Sigmoid()  # Values in [0,1]
        )

    def encode(self, x):
        h = self.encoder(x)
        #print(f"🔹 Encoder output shape before flattening: {h.shape}")
        #print(f"🔹 Input shape to fc_mu: {h.shape}")
        mu = self.fc_mu(h)
        log_var = self.fc_logvar(h)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc_decode(z)
        return self.decoder(h)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        recon_x = self.decode(z)
        return recon_x, mu, log_var, z

# Define the loss function
def loss_function(recon_x, x, mu, log_var):
    recon_loss = F.mse_loss(recon_x, x, reduction='sum') / x.shape[0]
    kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / x.shape[0]
    return recon_loss + kl_div, recon_loss, kl_div

# Train the VAE
def q2_a(train_data, test_data, dset_id, epochs=50, batch_size=128, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Convert and normalize data to torch tensors with shape (N, 3, 32, 32)
    transform = transforms.Compose([transforms.ToTensor()])
    train_data = torch.stack([transform(img) for img in train_data])
    test_data = torch.stack([transform(img) for img in test_data])

    train_loader = DataLoader(TensorDataset(train_data), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(test_data), batch_size=batch_size, shuffle=False)

    model = VAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    training_logs = []
    epoch_logs = []

    for epoch in range(epochs + 1):
        model.train()
        batch_losses = []
        print(f"\n🚀 Epoch [{epoch+1}/{epochs}] Started...")

        for batch_idx, (x,) in enumerate(train_loader):
            x = x.to(device)  # Use the entire batch (do not index further)
            optimizer.zero_grad()

            recon_x, mu, log_var, _ = model(x)
            loss, recon_loss, kl_div = loss_function(recon_x, x, mu, log_var)

            loss.backward()
            optimizer.step()

            batch_losses.append([loss.item(), recon_loss.item(), kl_div.item()])
            if batch_idx % 10 == 0:
                print(f"🟢 Epoch [{epoch+1}/{epochs}], Batch [{batch_idx}/{len(train_loader)}]")
                print(f"   🔹 ELBO Loss: {loss.item():.4f}")
                print(f"   🔹 Reconstruction Loss: {recon_loss.item():.4f}")
                print(f"   🔹 KL Divergence: {kl_div.item():.4f}\n")

        batch_losses = np.array(batch_losses)
        training_logs.append(batch_losses.mean(axis=0))

        # Evaluate on test data
        model.eval()
        with torch.no_grad():
            test_losses = []
            for batch_idx, (x,) in enumerate(test_loader):
                x = x.to(device)
                recon_x, mu, log_var, _ = model(x)
                loss, recon_loss, kl_div = loss_function(recon_x, x, mu, log_var)
                test_losses.append([loss.item(), recon_loss.item(), kl_div.item()])
            test_losses = np.array(test_losses)
            epoch_logs.append(test_losses.mean(axis=0))

        print(f"✅ Epoch [{epoch+1}/{epochs}] Completed: Avg ELBO = {training_logs[-1][0]:.4f}, Avg KL = {training_logs[-1][2]:.4f}")

    # Sampling from trained VAE
    with torch.no_grad():
        # 100 samples from the VAE
        z_samples = torch.randn(100, 16).to(device)
        samples = model.decode(z_samples).cpu() * 255  # shape: (100, 3, 32, 32)
        samples = samples.to(torch.uint8).permute(0, 2, 3, 1)  # convert to (N, H, W, C)

        # 50 real-image / reconstruction pairs
        real_images = test_data[:50].to(device)  # shape: (50, 3, 32, 32)
        reconstructed_images = model(real_images)[0].cpu() * 255  # shape: (50, 3, 32, 32)
        real_images = real_images.cpu().to(torch.uint8).permute(0, 2, 3, 1)
        reconstructed_images = reconstructed_images.to(torch.uint8).permute(0, 2, 3, 1)
        real_recon_pairs = torch.cat([real_images, reconstructed_images], dim=0)

        # 10 interpolations of 10 images (100 images total)
        interp_list = []
        for i in range(10):
            z1, z2 = model.encode(test_data[i].unsqueeze(0).to(device))
            alpha_vals = torch.linspace(0, 1, 10).to(device)
            interpolated_z = (1 - alpha_vals[:, None]) * z1 + alpha_vals[:, None] * z2
            interp_images = model.decode(interpolated_z).cpu() * 255  # (10, 3, 32, 32)
            interp_images = interp_images.to(torch.uint8).permute(0, 2, 3, 1)  # to (10, 32, 32, 3)
            interp_list.append(interp_images)
        interpolations = torch.cat(interp_list, dim=0)  # shape: (100, 32, 32, 3)

    return (
        np.array(training_logs),
        np.array(epoch_logs),
        samples.numpy(),
        real_recon_pairs.numpy(),
        interpolations.numpy()
    )

# Save results using the provided helper
q2_save_results('a', 2, q2_a)