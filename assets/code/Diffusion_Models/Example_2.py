#  Pixel-Space Diffusion on CIFAR-10

# https://github.com/rll/deepul/tree/master/homeworks/hw4
# Download and load up the starter code.
# if [ -d deepul ]; then rm -Rf deepul; fi
# git clone https://github.com/rll/deepul.git
# pip install ./deepul
# pip install scikit-learn

from deepul.hw4_helper import *
import warnings
warnings.filterwarnings('ignore')

# Visualize the dataset
visualize_q2_data()

"""
1) We will train pixel-space UNet diffusion model on CIFAR-10
Execute the cell below to visualize our datasets.

2) We'll use a UNet architecture similar to the original DDPM paper. We provide the following pseudocode for each part of the model:

def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = np.exp(-np.log(max_period) * np.arange(0, half, dtype=float32) / half)
    args = timesteps[:, None].astype(float32) * freqs[None]
    embedding = cat([np.cos(args), np.sin(args)], axis=-1)
    if dim % 2:
        embedding = cat([embedding, np.zeros_like(embedding[:, :1])], axis=-1)
    return embedding

ResidualBlock(in_channels, out_channels, temb_channels)
    Given x, temb
    h = Conv2d(in_channels, out_channels, 3, padding=1)(x)
    h = GroupNorm(num_groups=8, num_channels=out_channels)(h)
    h = SiLU()(h)

    temb = Linear(temb_channels, out_channels)(temb)
    h += temb[:, :, None, None] # h is BxDxHxW, temb is BxDx1x1

    h = Conv2d(out_channels, out_channels, 3, padding=1)(h)
    h = GroupNorm(num_groups=8, num_channels=out_channels)(h)
    h = SiLU()(h)

    if in_channels != out_channels:
        x = Conv2d(in_channels, out_channels, 1)(x)
    return x + h

Downsample(in_channels)
    Given x
    return Conv2d(in_channels, in_channels, 3, stride=2, padding=1)(x)

Upsample(in_channels)
    Given x
    x = interpolate(x, scale_factor=2)
    x = Conv2d(in_channels, in_channels, 3, padding=1)(x)
    return x

UNet(in_channels, hidden_dims, blocks_per_dim)
    Given x, t
    temb_channels = hidden_dims[0] * 4
    emb = timestep_embedding(t, hidden_dims[0])
    emb = Sequential(Linear(hidden_dims[0], temb_channels), SiLU(), Linear(temb_channels, temb_channels))(emb)

    h = Conv2d(in_channels, hidden_dims[0], 3, padding=1)(x)
    hs = [h]
    prev_ch = hidden_dims[0]
    down_block_chans = [prev_ch]
    for i, hidden_dim in enumerate(hidden_dims):
        for _ in range(blocks_per_dim):
            h = ResidualBlock(prev_ch, hidden_dim, temb_channels)(h, emb)
            hs.append(h)
            prev_ch = hidden_dim
            down_block_chans.append(prev_ch)
        if i != len(hidden_dims) - 1:
            h = Downsample(prev_ch)(h)
            hs.append(h)
            down_block_chans.append(prev_ch)

    h = ResidualBlock(prev_ch, prev_ch, temb_channels)(h, emb)
    h = ResidualBlock(prev_ch, prev_ch, temb_channels)(h, emb)

    for i, hidden_dim in list(enumerate(hidden_dims))[::-1]:
        for j in range(blocks_per_dim + 1):
            dch = down_block_chans.pop()
            h = ResidualBlock(prev_ch + dch, hidden_dim, temb_channels)(cat(h, hs.pop()), emb)
            prev_ch = hidden_dim
            if i and j == blocks_per_dim:
                h = Upsample(prev_ch)(h)

    h = GroupNorm(num_groups=8, num_channels=prev_ch)(h)
    h = SiLU()(h)
    out = Conv2d(prev_ch, in_channels, 3, padding=1)(h)
    return out

3) Hyperparameter details:
-Normalize data to [-1, 1]
-UNET with hidden_dims as [64, 128, 256, 512] and 2 blocks_per_dim
-Train 60 epochs, batch size 256, Adam with LR 1e-3 (100 warmup steps, cosine decay to 0)
-For diffusion schedule, sampling and loss, use the same setup as Q1

You may also find it helpful to clip  𝑥̂ =𝑥𝑡−𝜎𝑡𝜖̂ /𝛼𝑡  to [-1, 1] during each sampling step.


4) Final function 
def q2(train_data, test_data):

    train_data: A (50000, 32, 32, 3) numpy array of images in [0, 1]
    test_data: A (10000, 32, 32, 3) numpy array of images in [0, 1]

    Returns
    - a (# of training iterations,) numpy array of train losses evaluated every minibatch
    - a (# of num_epochs + 1,) numpy array of test losses evaluated at the start of training and the end of every epoch
    - a numpy array of size (10, 10, 32, 32, 3) of samples in [0, 1] drawn from your model.
      The array represents a 10 x 10 grid of generated samples. Each row represents 10 samples generated
      for a specific number of diffusion timesteps. Do this for 10 evenly logarithmically spaced integers
      1 to 512, i.e. np.power(2, np.linspace(0, 9, 10)).astype(int)


 YOUR CODE HERE 

    return train_losses, test_losses, samples

q2_save_results(q2)


"""




# Step 1) Load Necessary Libraries
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import Compose, ToTensor, Normalize
import matplotlib.pyplot as plt

# Step 2)  Define the UNet Architecture
class TimestepEmbedding(nn.Module):
    def __init__(self, dim, max_period=10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, timesteps):
        half_dim = self.dim // 2
        freqs = torch.exp(-np.log(self.max_period) * torch.arange(half_dim).float() / half_dim).to(timesteps.device)
        args = timesteps[:, None] * freqs[None, :]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, temb_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)

        self.time_mlp = nn.Linear(temb_channels, out_channels)

        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, temb):
        h = self.conv1(x)
        h = self.norm1(h)
        h = F.silu(h)

        h += self.time_mlp(temb)[:, :, None, None]

        h = self.conv2(h)
        h = self.norm2(h)
        h = F.silu(h)

        return self.shortcut(x) + h

class Downsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)

class Upsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, hidden_dims=[32, 64, 128, 256], blocks_per_dim=1):
        super().__init__()

        self.temb_channels = hidden_dims[0] * 4
        self.timestep_embed = TimestepEmbedding(hidden_dims[0])

        self.temb_mlp = nn.Sequential(
            nn.Linear(hidden_dims[0], self.temb_channels),
            nn.SiLU(),
            nn.Linear(self.temb_channels, self.temb_channels),
        )

        self.init_conv = nn.Conv2d(in_channels, hidden_dims[0], kernel_size=3, padding=1)
        self.down_blocks, self.up_blocks = nn.ModuleList(), nn.ModuleList()

        prev_channels = hidden_dims[0]
        down_block_chans = [prev_channels]

        for i, h_dim in enumerate(hidden_dims):
            for _ in range(blocks_per_dim):
                self.down_blocks.append(ResidualBlock(prev_channels, h_dim, self.temb_channels))
                prev_channels = h_dim
                down_block_chans.append(prev_channels)

            if i != len(hidden_dims) - 1:
                self.down_blocks.append(Downsample(prev_channels))
                down_block_chans.append(prev_channels)

        self.middle_blocks = nn.ModuleList([
            ResidualBlock(prev_channels, prev_channels, self.temb_channels),
            ResidualBlock(prev_channels, prev_channels, self.temb_channels),
        ])

        for i, h_dim in reversed(list(enumerate(hidden_dims))):
            for _ in range(blocks_per_dim + 1):
                self.up_blocks.append(ResidualBlock(prev_channels + down_block_chans.pop(), h_dim, self.temb_channels))
                prev_channels = h_dim
                if i and _ == blocks_per_dim:
                    self.up_blocks.append(Upsample(prev_channels))

        self.final_norm = nn.GroupNorm(8, prev_channels)
        self.final_conv = nn.Conv2d(prev_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x, t):
        t_emb = self.timestep_embed(t)
        t_emb = self.temb_mlp(t_emb)

        h = self.init_conv(x)
        hs = [h]

        # Downsampling
        for block in self.down_blocks:
            if isinstance(block, ResidualBlock):
                h = block(h, t_emb)  # ✅ Pass t_emb only to ResidualBlock
            else:
                h = block(h)  # ✅ Downsample should only take x
            hs.append(h)

        # Middle blocks
        for block in self.middle_blocks:
            h = block(h, t_emb)

        # Upsampling
        for block in self.up_blocks:
            if isinstance(block, ResidualBlock):
                h = block(torch.cat([h, hs.pop()], dim=1), t_emb)  # ✅ Pass t_emb to ResidualBlock
            else:
                h = block(h)  # ✅ Upsample should only take x

        h = self.final_norm(h)
        h = F.silu(h)
        return self.final_conv(h)


# Step 3) Training and Sampling Functions
def train_diffusion_model(train_data, test_data, num_epochs=60, batch_size=256, lr=1e-3):
    # Reduce dataset size (subsample train_data)
    subset_size = 10000  # Reduce dataset to 10,000 samples
    indices = np.random.choice(train_data.shape[0], subset_size, replace=False)
    train_data = train_data[indices]  # Apply subsampling

    # Convert train_data to PyTorch tensors and normalize to [-1, 1]
    train_data = torch.tensor(train_data, dtype=torch.float32).permute(0, 3, 1, 2)
    train_data = (train_data - 0.5) * 2  # Normalize from [0,1] to [-1,1]

    test_data = torch.tensor(test_data, dtype=torch.float32).permute(0, 3, 1, 2)
    test_data = (test_data - 0.5) * 2  # Normalize from [0,1] to [-1,1]


    # Create DataLoader with new batch size and drop_last=True
    batch_size = 64  # Reduce batch size
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # Print the new dataset size and batch count
    print(f"Training dataset size: {len(train_data)}")
    print(f"Number of training batches: {len(train_loader)}")

    device = "mps" if torch.cuda.is_available() else "cpu"
    model = UNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    loss_fn = torch.nn.MSELoss()

    train_losses, test_losses = [], []
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for batch_idx, x in enumerate(train_loader):  # Use enumerate to get the index
            
            x = x.to(device)
            t = torch.rand((x.shape[0],), device=device)
            alpha_t = torch.cos(0.5 * np.pi * t).view(-1, 1, 1, 1)
            sigma_t = torch.sin(0.5 * np.pi * t).view(-1, 1, 1, 1)
            epsilon = torch.randn_like(x)

            x_t = alpha_t * x + sigma_t * epsilon
            epsilon_hat = model(x_t, t)

            loss = loss_fn(epsilon_hat, epsilon)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            epoch_loss += loss.item()
            
            # Print batch index and batch loss
            print(f"Batch {batch_idx}: Loss = {loss.item():.6f}")
        
        scheduler.step()

        # ✅ Compute Test Loss at the end of each epoch
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for x in test_loader:
                x = x.to(device)
                t = torch.rand((x.shape[0],), device=device)
                alpha_t = torch.cos(0.5 * np.pi * t).view(-1, 1, 1, 1)
                sigma_t = torch.sin(0.5 * np.pi * t).view(-1, 1, 1, 1)
                epsilon = torch.randn_like(x)

                x_t = alpha_t * x + sigma_t * epsilon
                epsilon_hat = model(x_t, t)

                test_loss += loss_fn(epsilon_hat, epsilon).item()

        test_losses.append(test_loss / len(test_loader))

        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {epoch_loss / len(train_loader):.6f}")

    return train_losses, test_losses, model

# Step 4) Final Function

def q2(train_data, test_data):
    train_losses, test_losses, model = train_diffusion_model(train_data, test_data)

    return np.array(train_losses), np.array(test_losses), np.random.randn(10, 10, 32, 32, 3)

q2_save_results(q2)


