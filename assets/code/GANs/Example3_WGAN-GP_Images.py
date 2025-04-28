# https://github.com/rll/deepul/blob/master/homeworks/hw3/hw3.ipynb
# We will train a GAN on CIFAR-10. Execute the cell below to visualize the dataset.

# Download and load up the starter code.
# if [ -d deepul ]; then rm -Rf deepul; fi
# git clone https://github.com/rll/deepul.git
# pip install ./deepul
# pip install scikit-learn

"""
We'll use the CIFAR-10 architecture from the SN-GAN paper with  z \in R^128 , with  z~N(0,𝐼128) . 
Instead of upsampling via transposed convolutions and downsampling via pooling or striding, we'll use these DepthToSpace 
and SpaceToDepth methods for changing the spatial configuration of our hidden states.

class DepthToSpace(nn.Module):
    def __init__(self, block_size):
        super().__init__()
        self.block_size = block_size
        self.block_size_sq = block_size * block_size

    def forward(self, input):
        output = input.permute(0, 2, 3, 1)
        (batch_size, d_height, d_width, d_depth) = output.size()
        s_depth = int(d_depth / self.block_size_sq)
        s_width = int(d_width * self.block_size)
        s_height = int(d_height * self.block_size)
        t_1 = output.reshape(batch_size, d_height, d_width, self.block_size_sq, s_depth)
        spl = t_1.split(self.block_size, 3)
        stack = [t_t.reshape(batch_size, d_height, s_width, s_depth) for t_t in spl]
        output = torch.stack(stack, 0).transpose(0, 1).permute(0, 2, 1, 3, 4).reshape(batch_size, s_height, s_width,
                                                                                      s_depth)
        output = output.permute(0, 3, 1, 2)
        return output


class SpaceToDepth(nn.Module):
    def __init__(self, block_size):
        super().__init__()
        self.block_size = block_size
        self.block_size_sq = block_size * block_size

    def forward(self, input):
        output = input.permute(0, 2, 3, 1)
        (batch_size, s_height, s_width, s_depth) = output.size()
        d_depth = s_depth * self.block_size_sq
        d_width = int(s_width / self.block_size)
        d_height = int(s_height / self.block_size)
        t_1 = output.split(self.block_size, 2)
        stack = [t_t.reshape(batch_size, d_height, d_depth) for t_t in t_1]
        output = torch.stack(stack, 1)
        output = output.permute(0, 2, 1, 3)
        output = output.permute(0, 3, 1, 2)
        return output

# Spatial Upsampling with Nearest Neighbors
Upsample_Conv2d(in_dim, out_dim, kernel_size=(3, 3), stride=1, padding=1):
    x = torch.cat([x, x, x, x], dim=1)
    DepthToSpace(block_size=2)
    Conv2d(in_dim, out_dim, kernel_size, stride=stride, padding=padding)


# Spatial Downsampling with Spatial Mean Pooling
Downsample_Conv2d(in_dim, out_dim, kernel_size=(3, 3), stride=1, padding=1):
        SpaceToDepth(2)
        torch.sum(x.chunk(4, dim=1)) / 4.0
        nn.Conv2d(in_dim, out_dim, kernel_size,
                              stride=stride, padding=padding, bias=bias)


                             
                              
Here's pseudocode for how we'll implement a ResBlockUp, used in the generator:   
ResnetBlockUp(x, in_dim, kernel_size=(3, 3), n_filters=256):
    _x = x
    _x = nn.BatchNorm2d(in_dim)(_x)
    _x = nn.ReLU()(_x)
    _x = nn.Conv2d(in_dim, n_filters, kernel_size, padding=1)(_x)
    _x = nn.BatchNorm2d(n_filters)(_x)
    _x = nn.ReLU()(_x)
    residual = Upsample_Conv2d(n_filters, n_filters, kernel_size, padding=1)(_x)
    shortcut = Upsample_Conv2d(in_dim, n_filters, kernel_size=(1, 1), padding=0)(x)
    return residual + shortcut

    
The ResBlockDown module is similar, except it uses Downsample_Conv2d and omits the BatchNorm.
Finally, here's the architecture for the generator:

def Generator(*, n_samples=1024, n_filters=128):
    z = Normal(0, 1)([n_samples, 128])
    nn.Linear(128, 4*4*256)
    reshape output of linear layer
    ResnetBlockUp(in_dim=256, n_filters=n_filters),
    ResnetBlockUp(in_dim=n_filters, n_filters=n_filters),
    ResnetBlockUp(in_dim=n_filters, n_filters=n_filters),
    nn.BatchNorm2d(n_filters),
    nn.ReLU(),
    nn.Conv2d(n_filters, 3, kernel_size=(3, 3), padding=1),
    nn.Tanh()

The discriminator (no BatchNorm!):
def Discriminator(*):
    def __init__(self, n_filters=128):
        ResnetBlockDown(3, n_filters=n_filters),
        ResnetBlockDown(128, n_filters=n_filters),
        ResBlock(n_filters, n_filters=n_filters),
        ResBlock(n_filters, n_filters=n_filters),
        nn.ReLU()
        global sum pooling
        nn.Linear(128, 1)


Hyperparameters

We'll implement WGAN-GP, which uses a gradient penalty to regularize the discriminator. Use the Adam optimizer with 𝛼=2e^(-4)
, 𝛽1=0 ,  𝛽2=0.9 ,  𝜆=10 ,  n_critic=5 . 
Use a batch size of 256 and n_filters=128 within the ResBlocks. 
Train for at least 25000 gradient steps, with the learning rate linearly annealed to 0 over training.

Deliverables:
-Inception score (CIFAR-10 version) of the final model. We provide a utility that will automatically do this for you.
-Fréchet inception distance 
-Discriminator loss across training
-100 samples.


def q2(train_data):
   
    train_data: An (n_train, 3, 32, 32) numpy array of CIFAR-10 images with values in [0, 1]

    Returns
    - a (# of training iterations,) numpy array of WGAN critic train losses evaluated every minibatch
    - a (1000, 32, 32, 3) numpy array of samples from your model in [0, 1]. 
        The first 100 will be displayed, and the rest will be used to calculate the Inception score. 
   
   
   YOUR CODE HERE 

    return losses, samples


q2_save_results(q2)
"""


from deepul.hw3_helper import visualize_q2_data, q2_save_results
import deepul.pytorch_util as ptu
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Visualize the dataset (provided helper function)
visualize_q2_data()

# Determine device: use MPS if available, else CPU.
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print("Using device:", device)
ptu.device = device

# --- Monkey-patch torch.load to always map weights to our device ---
orig_torch_load = torch.load
def my_torch_load(*args, **kwargs):
    if "map_location" not in kwargs:
         kwargs["map_location"] = device
    return orig_torch_load(*args, **kwargs)
torch.load = my_torch_load
# --- End monkey-patch ---

########################################
# 1. DepthToSpace & SpaceToDepth using PixelShuffle/PixelUnshuffle
########################################
class DepthToSpace(nn.Module):
    """Upsampling using PixelShuffle."""
    def __init__(self, block_size):
        super().__init__()
        self.pixel_shuffle = nn.PixelShuffle(block_size)
    def forward(self, x):
        return self.pixel_shuffle(x)

class SpaceToDepth(nn.Module):
    """Downsampling using PixelUnshuffle."""
    def __init__(self, block_size):
        super().__init__()
        self.pixel_unshuffle = nn.PixelUnshuffle(block_size)
    def forward(self, x):
        return self.pixel_unshuffle(x)

########################################
# 2. UpsampleConv2d & DownsampleConv2d
########################################
class UpsampleConv2d(nn.Module):
    """
    Upsample by repeating channels 4x, then use PixelShuffle,
    followed by a convolution.
    """
    def __init__(self, in_dim, out_dim, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.upsample = DepthToSpace(block_size=2)
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size, stride=stride, padding=padding)
    def forward(self, x):
        x = x.repeat(1, 4, 1, 1)
        x = self.upsample(x)
        return self.conv(x)

class DownsampleConv2d(nn.Module):
    """
    Downsample using PixelUnshuffle, then a convolution.
    """
    def __init__(self, in_dim, out_dim, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.downsample = SpaceToDepth(block_size=2)
        self.conv = nn.Conv2d(in_dim * 4, out_dim, kernel_size, stride=stride, padding=padding)
    def forward(self, x):
        x = self.downsample(x)
        return self.conv(x)

########################################
# 3. ResBlockUp (Generator) & ResBlockDown (Discriminator)
########################################
class ResBlockUp(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(in_dim),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=False),
        )
        self.residual = UpsampleConv2d(out_dim, out_dim, kernel_size=3, padding=1)
        self.shortcut = UpsampleConv2d(in_dim, out_dim, kernel_size=1, padding=0)
    def forward(self, x):
        _x = self.block(x)
        return self.residual(_x) + self.shortcut(x)

class ResBlockDown(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
        )
        self.residual = DownsampleConv2d(out_dim, out_dim, kernel_size=3, padding=1)
        self.shortcut = DownsampleConv2d(in_dim, out_dim, kernel_size=1, padding=0)
    def forward(self, x):
        _x = self.block(x)
        return self.residual(_x) + self.shortcut(x)

########################################
# 4. Basic ResBlock (Discriminator)
########################################
class ResBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.shortcut_conv = None
        if in_dim != out_dim:
            self.shortcut_conv = nn.Conv2d(in_dim, out_dim, kernel_size=1, padding=0)
        self.block = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),
        )
    def forward(self, x):
        shortcut = x if self.shortcut_conv is None else self.shortcut_conv(x)
        return self.block(x) + shortcut

########################################
# 5. Generator & Discriminator
########################################
class Generator(nn.Module):
    def __init__(self, z_dim=128, n_filters=128):
        super().__init__()
        self.z_dim = z_dim
        self.linear = nn.Linear(z_dim, 4 * 4 * 256)
        self.rb1 = ResBlockUp(256, n_filters)
        self.rb2 = ResBlockUp(n_filters, n_filters)
        self.rb3 = ResBlockUp(n_filters, n_filters)
        self.bn = nn.BatchNorm2d(n_filters)
        self.conv_final = nn.Conv2d(n_filters, 3, kernel_size=3, padding=1)
    def forward(self, z):
        out = self.linear(z)
        out = out.contiguous().reshape(-1, 256, 4, 4)
        out = self.rb1(out)
        out = self.rb2(out)
        out = self.rb3(out)
        out = self.bn(out)
        out = F.relu(out, inplace=False)
        out = self.conv_final(out)
        out = torch.tanh(out)
        return out

class Discriminator(nn.Module):
    def __init__(self, n_filters=128):
        super().__init__()
        self.rbdown1 = ResBlockDown(3, n_filters)
        self.rbdown2 = ResBlockDown(n_filters, n_filters)
        self.rb3 = ResBlock(n_filters, n_filters)
        self.rb4 = ResBlock(n_filters, n_filters)
        self.relu = nn.ReLU(inplace=True)
        self.linear = nn.Linear(n_filters, 1)
    def forward(self, x):
        out = self.rbdown1(x)
        out = self.rbdown2(out)
        out = self.rb3(out)
        out = self.rb4(out)
        out = self.relu(out)
        out = out.sum(dim=[2, 3])
        out = self.linear(out)
        return out

########################################
# 6. WGANTrainer (WGAN-GP) with detailed prints
########################################
class WGANTrainer:
    def __init__(
        self,
        train_data,
        z_dim=128,
        n_filters=128,
        batch_size=256,
        total_steps=25000,
        lr=2e-4,
        beta1=0.0,
        beta2=0.9,
        gp_lambda=10,
        n_critic=5,
        device=device
    ):
        """
        train_data: (N, 3, 32, 32) numpy array in [0, 1].
        Images are shifted to [-1, 1] during training.
        """
        self.device = device
        self.z_dim = z_dim
        self.n_filters = n_filters
        self.batch_size = batch_size
        self.total_steps = total_steps
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.gp_lambda = gp_lambda
        self.n_critic = n_critic

        dataset = TensorDataset(torch.from_numpy(train_data).float())
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        self.data_iter = iter(self.dataloader)

        self.epoch = 0
        self.batch_count = 0

        self.G = Generator(z_dim=z_dim, n_filters=n_filters).to(device)
        self.D = Discriminator(n_filters=n_filters).to(device)

        self.opt_G = optim.Adam(self.G.parameters(), lr=lr, betas=(beta1, beta2))
        self.opt_D = optim.Adam(self.D.parameters(), lr=lr, betas=(beta1, beta2))

        self.disc_losses = []
        self.step = 0

    def gradient_penalty(self, real_imgs, fake_imgs):
        batch_size = real_imgs.size(0)
        alpha = torch.rand(batch_size, 1, 1, 1, device=self.device)
        alpha = alpha.expand_as(real_imgs)
        interp = alpha * real_imgs + (1 - alpha) * fake_imgs
        interp.requires_grad_(True)
        d_out = self.D(interp)
        grad_outputs = torch.ones_like(d_out, device=self.device)
        grads = torch.autograd.grad(
            outputs=d_out,
            inputs=interp,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        grads = grads.contiguous().reshape(batch_size, -1)
        gp = ((grads.norm(2, dim=1) - 1) ** 2).mean()
        return self.gp_lambda * gp

    def get_lr(self, step):
        return self.lr * max(0.0, 1.0 - step / self.total_steps)

    def train_step(self):
        d_loss_val = None
        for i in range(self.n_critic):
            try:
                real_batch = next(self.data_iter)[0].to(self.device)
                self.batch_count += 1
            except StopIteration:
                self.epoch += 1
                print(f"Epoch {self.epoch} completed.")
                self.data_iter = iter(self.dataloader)
                real_batch = next(self.data_iter)[0].to(self.device)
                self.batch_count = 1

            current_bs = real_batch.size(0)
            real_batch = real_batch * 2 - 1.0

            for param_group in self.opt_D.param_groups:
                param_group['lr'] = self.get_lr(self.step)

            z = torch.randn(current_bs, self.z_dim, device=self.device)
            fake_imgs = self.G(z).detach()

            d_real = self.D(real_batch)
            d_fake = self.D(fake_imgs)
            d_loss = -d_real.mean() + d_fake.mean()
            gp = self.gradient_penalty(real_batch, fake_imgs)
            d_loss_total = d_loss + gp

            self.opt_D.zero_grad()
            d_loss_total.backward()
            self.opt_D.step()

            d_loss_val = d_loss.item()
            self.disc_losses.append(d_loss_val)
            print(f"Step {self.step}/{self.total_steps} | Epoch {self.epoch} Batch {self.batch_count} | D_loss = {d_loss_val:.4f}")

        for param_group in self.opt_G.param_groups:
            param_group['lr'] = self.get_lr(self.step)

        z = torch.randn(current_bs, self.z_dim, device=self.device)
        fake_imgs = self.G(z)
        g_loss = -self.D(fake_imgs).mean()

        self.opt_G.zero_grad()
        g_loss.backward()
        self.opt_G.step()

        self.step += 1
        print(f"Step {self.step}/{self.total_steps} | Epoch {self.epoch} Batch {self.batch_count} | G_loss = {g_loss.item():.4f}")
        return d_loss_val, g_loss.item()

    def train(self):
        while self.step < self.total_steps:
            d_loss_val, g_loss_val = self.train_step()
            if self.step % 100 == 0:
                print(f"--> Step {self.step}/{self.total_steps} | Latest D_loss = {d_loss_val:.4f} | G_loss = {g_loss_val:.4f}")
        return np.array(self.disc_losses)

########################################
# 7. The q2 Function
########################################
def q2(train_data):
    """
    train_data: (n_train, 3, 32, 32) numpy array of CIFAR-10 images in [0, 1].
    Returns:
      - disc_losses: (num_iterations,) numpy array of WGAN critic losses.
      - samples: (1000, 32, 32, 3) numpy array of samples from the model in [0, 1].
    """
    z_dim = 128
    n_filters = 128
    batch_size = 256
    total_steps = 25000
    lr = 2e-4
    beta1 = 0.0
    beta2 = 0.9
    gp_lambda = 10
    n_critic = 5

    trainer = WGANTrainer(
        train_data,
        z_dim=z_dim,
        n_filters=n_filters,
        batch_size=batch_size,
        total_steps=total_steps,
        lr=lr,
        beta1=beta1,
        beta2=beta2,
        gp_lambda=gp_lambda,
        n_critic=n_critic,
        device=device
    )
    disc_losses = trainer.train()

    trainer.G.eval()
    with torch.no_grad():
        z = torch.randn(1000, z_dim, device=device)
        fake_imgs = trainer.G(z)
        fake_imgs = (fake_imgs + 1) / 2
        fake_imgs = fake_imgs.clamp(0, 1)
        fake_imgs = fake_imgs.permute(0, 2, 3, 1).cpu().numpy()

    return disc_losses, fake_imgs

q2_save_results(q2)