# https://github.com/rll/deepul/blob/master/homeworks/hw3/hw3.ipynb
# GAN original formulation
# we will train different variants of GANs on an easy 1D dataset.

# Download and load up the starter code.
# if [ -d deepul ]; then rm -Rf deepul; fi
# git clone https://github.com/rll/deepul.git
# pip install ./deepul
# pip install scikit-learn

""""
In this part, we'll train our generator and discriminator via the original minimax GAN objective:
min_G max_D: E_{x ~ p_data}[log D(x)] + E_{z ~ p_z}[log(1 - D(G(z)))]

Use an MLP for both your generator and your discriminator, and train until the generated distribution resembles the target distribution.

-3 layers
-128 hidden dim
-LeakyReLU nonlinearities with negative_slope=0.2"
"""

"""
def q1_a(train_data):
   
    train_data: An (20000, 1) numpy array of floats in [-1, 1]

    Returns
    - a (# of training iterations,) numpy array of discriminator losses evaluated every minibatch
    - a numpy array of size (5000,) of samples drawn from your model at epoch #1
    - a numpy array of size (1000,) linearly spaced from [-1, 1]; hint: np.linspace
    - a numpy array of size (1000,), corresponding to the discriminator output (after sigmoid) 
        at each location in the previous array at epoch #1

    - a numpy array of size (5000,) of samples drawn from your model at the end of training
    - a numpy array of size (1000,) linearly spaced from [-1, 1]; hint: np.linspace
    - a numpy array of size (1000,), corresponding to the discriminator output (after sigmoid) 
        at each location in the previous array at the end of training
 

  Main Code 
  

 q1_save_results('a', q1_a) 
  """


from deepul.hw3_helper import *
import deepul.pytorch_util as ptu
import warnings
warnings.filterwarnings('ignore')
ptu.set_gpu_mode(True)

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
from scipy.stats import norm
from tqdm import trange, tqdm_notebook

# Visualize the dataset (provided helper function)
visualize_q1_dataset()


# Determine the device: use MPS if available, else CPU.
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print("Using device:", device)

# Define the Generator network
class Generator(nn.Module):
    def __init__(self, noise_dim=1, hidden_dim=128, output_dim=1):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(noise_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, z):
        return self.model(z)

# Define the Discriminator network
class Discriminator(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=128):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x):
        return self.model(x)  # returns logits; apply sigmoid externally when needed

# Define a training class that encapsulates the training process and loss functions
class GANTrainer:
    def __init__(self, train_data, noise_dim=1, hidden_dim=128, batch_size=128, num_epochs=50, lr=1e-3, device=device):
        self.device = device
        # Convert the training data to a torch tensor and move to device
        self.train_data = torch.from_numpy(train_data).float().to(self.device)
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.noise_dim = noise_dim
        
        # Instantiate the generator and discriminator
        self.G = Generator(noise_dim=noise_dim, hidden_dim=hidden_dim, output_dim=1).to(self.device)
        self.D = Discriminator(input_dim=1, hidden_dim=hidden_dim).to(self.device)
        
        # Create optimizers
        self.optimizer_D = optim.Adam(self.D.parameters(), lr=lr)
        self.optimizer_G = optim.Adam(self.G.parameters(), lr=lr)
        
        # Create a DataLoader for the training data
        dataset = torch.utils.data.TensorDataset(self.train_data)
        self.dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # List to record discriminator losses for every minibatch
        self.disc_losses = []
        
        # Placeholders for results at epoch 1
        self.epoch1_samples = None
        self.disc_outputs_epoch1 = None
        
    def discriminator_loss(self, real_data, fake_data):
        # Compute discriminator outputs for real and fake data
        D_real = self.D(real_data)
        D_fake = self.D(fake_data)
        # Apply sigmoid to get probabilities
        D_real_prob = torch.sigmoid(D_real)
        D_fake_prob = torch.sigmoid(D_fake)
        # Compute the log-likelihoods for real and fake data
        loss_real = torch.log(D_real_prob + 1e-8).mean()
        loss_fake = torch.log(1 - D_fake_prob + 1e-8).mean()
        # The discriminator loss is the negative of the sum (since we maximize the log-likelihoods)
        loss_D = -(loss_real + loss_fake)
        return loss_D
        
    def generator_loss(self, fake_data):
        # For the generator, compute log(1 - D(G(z))) on the fake data
        D_fake = self.D(fake_data)
        D_fake_prob = torch.sigmoid(D_fake)
        loss_G = -torch.log(1 - D_fake_prob + 1e-8).mean()
        return loss_G
        
    def train(self):
        # Create a linspace for evaluating discriminator outputs (used at epoch 1 and final evaluation)
        x_lin = np.linspace(-1, 1, 1000)
        
        for epoch in range(self.num_epochs):
            epoch_D_loss = 0.0
            epoch_G_loss = 0.0
            num_batches = 0
            
            for batch_idx, batch in enumerate(self.dataloader):
                batch_real = batch[0]
                current_batch_size = batch_real.size(0)
                
                ## Train the Discriminator
                self.optimizer_D.zero_grad()
                noise = torch.randn(current_batch_size, self.noise_dim, device=self.device)
                fake_data = self.G(noise)
                loss_D = self.discriminator_loss(batch_real, fake_data)
                loss_D.backward()
                self.optimizer_D.step()
                
                ## Train the Generator
                self.optimizer_G.zero_grad()
                noise = torch.randn(current_batch_size, self.noise_dim, device=self.device)
                fake_data = self.G(noise)
                loss_G = self.generator_loss(fake_data)
                loss_G.backward()
                self.optimizer_G.step()
                
                # Record the discriminator loss for this minibatch
                self.disc_losses.append(loss_D.item())
                
                # Update epoch loss accumulators
                epoch_D_loss += loss_D.item()
                epoch_G_loss += loss_G.item()
                num_batches += 1
                
                # Print the training process: epoch, batch, discriminator loss, and generator loss
                print(f"Epoch {epoch+1}/{self.num_epochs} Batch {batch_idx+1}/{len(self.dataloader)}: "
                      f"D_loss = {loss_D.item():.4f}, G_loss = {loss_G.item():.4f}")
            
            # Print average losses for the epoch
            avg_D_loss = epoch_D_loss / num_batches
            avg_G_loss = epoch_G_loss / num_batches
            print(f"--> Epoch {epoch+1} Average: D_loss = {avg_D_loss:.4f}, G_loss = {avg_G_loss:.4f}\n")
            
            # At the end of the first epoch, record generator samples and discriminator outputs
            if epoch == 0:
                with torch.no_grad():
                    noise_epoch1 = torch.randn(5000, self.noise_dim, device=self.device)
                    samples = self.G(noise_epoch1)
                    self.epoch1_samples = samples.cpu().numpy().squeeze()
                    
                    # Evaluate discriminator on a linspace from -1 to 1
                    x_tensor = torch.tensor(x_lin, dtype=torch.float32, device=self.device).unsqueeze(1)
                    disc_out = torch.sigmoid(self.D(x_tensor))
                    self.disc_outputs_epoch1 = disc_out.cpu().numpy().squeeze()
        
        # After training, record final samples and discriminator outputs
        with torch.no_grad():
            noise_final = torch.randn(5000, self.noise_dim, device=self.device)
            final_samples = self.G(noise_final)
            final_samples_np = final_samples.cpu().numpy().squeeze()
            
            x_tensor_final = torch.tensor(x_lin, dtype=torch.float32, device=self.device).unsqueeze(1)
            disc_out_final = torch.sigmoid(self.D(x_tensor_final))
            disc_outputs_final = disc_out_final.cpu().numpy().squeeze()
        
        # Convert the list of discriminator losses to a numpy array
        disc_losses_np = np.array(self.disc_losses)
        
        return (disc_losses_np, 
                self.epoch1_samples, 
                x_lin, 
                self.disc_outputs_epoch1, 
                final_samples_np, 
                x_lin, 
                disc_outputs_final)

# Define the function q1_a that creates the trainer and runs the training process.
def q1_a(train_data):
    trainer = GANTrainer(train_data)
    return trainer.train()

# Save results using the provided helper function.
q1_save_results('a', q1_a)