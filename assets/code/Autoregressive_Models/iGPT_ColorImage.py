# https://github.com/rll/deepul/tree/master/homeworks/hw1

# Autoregressive Transformer on Color Images

# Download and load up the starter code.
# if [ -d deepul ]; then rm -Rf deepul; fi
# git clone https://github.com/rll/deepul.git
# pip install ./deepul
# pip install scikit-learn

""""
Now, implement an iGPT that models color. In order to reduce the length of token sequences, iGPT models each RGB pixel as a single token. This effectively reduces the context length from HWC to just H*W. iGPT does this through a k-means clustering approach. Because our images only each can only take on 4 values (2 bits) per channel, we can represent each pixel with 64 values (6 bits). Convert the dataset into an image of tokens and train iGPT on the colored shapes and MNIST dataset.

Checkout the iGPT paper for more details: Generative Pretraining from Pixels

Training times and hyperparameter settings should be the same as part (a), except train for longer (15 epochs)

You will provide these deliverables

Over the course of training, record the average negative log-likelihood (nats / dim) of the training data (per minibatch) and test data (for your entire test set). Code is provided that automatically plots the training curves.
Report the final test set performance of your final model
100 samples from the final trained model

def q3_b(train_data, test_data, image_shape, dset_id):

  train_data: A (n_train, H, W, C) uint8 numpy array of color images with values in {0, 1, 2, 3}
  test_data: A (n_test, H, W, C) uint8 numpy array of color images with values in {0, 1, 2, 3}
  image_shape: (H, W, C), height, width, and # of channels of the image
  dset_id: An identifying number of which dataset is given (1 or 2). Most likely
           used to set different hyperparameters for different datasets

  Returns
  - a (# of training iterations,) numpy array of train_losses evaluated every minibatch
  - a (# of epochs + 1,) numpy array of test_losses evaluated once at initialization and after each epoch
  - a numpy array of size (100, H, W, C) of samples with values in {0, 1, 2, 3}
 
  return train_losses, test_losses, samples

  
Once you've implemented q3_b, execute the cells below to visualize and save your results

q3ab_save_results(1, 'b', q3_b)
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from deepul.hw1_helper import (
    visualize_q1_data,
    q1_sample_data_1,
    q1_sample_data_2,
    q1_save_results,
    q2a_save_results,
    q2b_save_results,
    visualize_q2a_data,
    visualize_q2b_data,
    q3ab_save_results,
    q3c_save_results,
    q4a_save_results,
    q4b_save_results,
    visualize_q5_data,
    q5a_save_results,
    visualize_q6_data,
    q6a_save_results,
)


# -------------------------
# (Optional) Debug Visualization
# -------------------------
def debug_visualize_color_images(data, num_images=16):
    """
    Visualizes a random subset of color images from the training set.
    
    Args:
      data: A numpy array of shape (n, H, W, 3) with values in {0, 1, 2, 3}.
      num_images: The number of images to visualize.
    """
    indices = np.random.choice(len(data), size=num_images, replace=False)
    sample = data[indices]
    # Scale pixel values: 0->0, 3->255.
    sample_scaled = sample * 85  
    grid_size = int(np.ceil(np.sqrt(num_images)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(2 * grid_size, 2 * grid_size))
    axes = axes.flatten()
    for i in range(num_images):
        axes[i].imshow(sample_scaled[i])
        axes[i].axis('off')
    for i in range(num_images, len(axes)):
        axes[i].axis('off')
    plt.suptitle("Debug Visualization of Color Training Data")
    plt.show()

# -------------------------
# Data Processing Components for Color Images
# -------------------------
class ColorImageSequenceDataset(Dataset):
    """
    Converts a color image (H, W, 3) with pixel values in {0,1,2,3} into a sequence.
    Each pixel is encoded as:
       token = R * 16 + G * 4 + B
    yielding values in 0...63, then shifted by +1 (so token 0 is reserved as <bos>).
    """
    def __init__(self, data):
        # data: numpy array of shape (n, H, W, 3)
        self.data = data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        img = self.data[idx]  # shape (H, W, 3)
        # Compute token for each pixel
        tokens = img[..., 0] * 16 + img[..., 1] * 4 + img[..., 2]  # values in 0...63
        tokens = tokens.flatten()  # shape (H*W,)
        tokens = tokens.astype(np.int64) + 1  # shift: now in 1...64, with 0 reserved for <bos>
        seq = np.concatenate(([0], tokens))
        return torch.tensor(seq, dtype=torch.long)

# -------------------------
# Transformer Model Components
# -------------------------
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_head = d_model // n_heads
        self.n_heads = n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
    
    def forward(self, x, mask):
        B, T, C = x.size()
        qkv = self.qkv(x)  # (B, T, 3*d_model)
        qkv = qkv.reshape(B, T, self.n_heads, 3 * self.d_head)
        qkv = qkv.permute(0, 2, 1, 3)  # (B, n_heads, T, 3*d_head)
        q, k, v = torch.chunk(qkv, 3, dim=-1)  # each (B, n_heads, T, d_head)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        attn_scores = attn_scores.masked_fill(~mask, float('-inf'))
        attn = torch.softmax(attn_scores, dim=-1)
        out = torch.matmul(attn, v)  # (B, n_heads, T, d_head)
        out = out.transpose(1, 2).reshape(B, T, C)
        out = self.out_proj(out)
        return out

class FeedForward(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.fc1 = nn.Linear(d_model, 4*d_model)
        self.fc2 = nn.Linear(4*d_model, d_model)
        self.act = nn.GELU()
    
    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.mha = MultiHeadSelfAttention(d_model, n_heads)
        self.ff = FeedForward(d_model)
    
    def forward(self, x, mask):
        x = x + self.mha(self.ln1(x), mask)
        x = x + self.ff(self.ln2(x))
        return x

class TransformerModel(nn.Module):
    def __init__(self, seq_length, vocab_size, d_model, n_heads, n_layers):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(seq_length, d_model))
        self.layers = nn.ModuleList([TransformerBlock(d_model, n_heads) for _ in range(n_layers)])
        self.ln_final = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        self.seq_length = seq_length
    
    def forward(self, x):
        B, T = x.size()
        x = self.token_embedding(x) + self.pos_embedding[:T].unsqueeze(0)
        mask = torch.tril(torch.ones(T, T, device=x.device)).bool()
        for layer in self.layers:
            x = layer(x, mask)
        x = self.ln_final(x)
        logits = self.head(x)
        return logits

# -------------------------
# Trainer Class
# -------------------------
class Trainer:
    def __init__(self, model, optimizer, scheduler, criterion,
                 train_loader, test_loader, device, vocab_size, seq_length):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.vocab_size = vocab_size
        self.seq_length = seq_length
    
    def train_epoch(self, epoch):
        self.model.train()
        epoch_losses = []
        for batch_idx, batch in enumerate(self.train_loader):
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            logits = self.model(batch)  # (B, T, vocab_size)
            # Compute loss on tokens 1: (the image pixels)
            logits = logits[:, 1:, :]
            targets = batch[:, 1:]
            loss = self.criterion(logits.reshape(-1, self.vocab_size), targets.reshape(-1))
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            epoch_losses.append(loss.item())
            if (batch_idx + 1) % 10 == 0:
                print(f"[Epoch {epoch+1}, Batch {batch_idx+1}/{len(self.train_loader)}] loss = {loss.item():.4f}")
        return epoch_losses
    
    def evaluate(self):
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        with torch.no_grad():
            for batch in self.test_loader:
                batch = batch.to(self.device)
                logits = self.model(batch)
                logits = logits[:, 1:, :]
                targets = batch[:, 1:]
                loss = self.criterion(logits.reshape(-1, self.vocab_size), targets.reshape(-1))
                total_loss += loss.item() * targets.numel()
                total_tokens += targets.numel()
        return total_loss / total_tokens
    
    def sample(self, num_samples, image_shape, temperature=1.0):
        self.model.eval()
        samples = []
        with torch.no_grad():
            for sidx in range(num_samples):
                generated = torch.tensor([[0]], dtype=torch.long, device=self.device)
                for _ in range(self.seq_length - 1):
                    logits = self.model(generated)
                    logits = logits[:, -1, :]
                    logits[:, 0] = -float('inf')
                    probs = torch.softmax(logits / temperature, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    generated = torch.cat((generated, next_token), dim=1)
                if sidx < 3:
                    tokens = generated.squeeze().cpu().numpy()
                    print(f"[Sampling Debug] Sample {sidx} - Raw token sequence: {tokens}")
                    print(f"[Sampling Debug] Unique tokens before mapping: {np.unique(tokens)}")
                img_tokens = generated[0, 1:] - 1
                H, W, _ = image_shape
                img = img_tokens.reshape(H, W, 1).cpu().numpy().astype(np.uint8)
                samples.append(img)
                if sidx < 3:
                    print(f"[Sampling Debug] Sample {sidx} - Unique token values: {np.unique(img)}")
        return np.stack(samples, axis=0)

# -------------------------
# q3_b Function Implementation
# -------------------------
def q3_b(train_data, test_data, image_shape, dset_id):
    """
    Trains an autoregressive transformer (iGPT) to model color images.
    
    Args:
      train_data: (n_train, H, W, 3) uint8 numpy array with values in {0,1,2,3}
      test_data:  (n_test, H, W, 3) uint8 numpy array with values in {0,1,2,3}
      image_shape: (H, W, 3)
      dset_id: Dataset id (1 or 2)
    
    Returns:
      train_losses, test_losses, samples
    """
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    H, W, C = image_shape
    seq_length = 1 + H * W
    vocab_size = 64 + 1  # 0 is <bos>; tokens 1..64 for pixel colors.
    
    print("[DEBUG] train_data shape:", train_data.shape)
    print("[DEBUG] unique pixel values in train_data:", np.unique(train_data))
    print("[DEBUG] test_data shape:", test_data.shape)
    print("[DEBUG] unique pixel values in test_data:", np.unique(test_data))
    # Optionally: debug_visualize_images(train_data, num_images=16)
    
    train_dataset = ColorImageSequenceDataset(train_data)
    test_dataset = ColorImageSequenceDataset(test_data)
    batch_size = 64
    num_epochs = 15
    lr = 1e-3
    warmup_steps = 1000
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    debug_visualize_color_images(train_data, num_images=16)

    model = TransformerModel(seq_length=seq_length, vocab_size=vocab_size,
                             d_model=128, n_heads=4, n_layers=2).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    total_steps = num_epochs * len(train_loader)
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Compute class weights:
    # Map each pixel (with values 0..3) to a token: token = R*16 + G*4 + B, values in 0...63.
    pixels = train_data[..., 0] * 16 + train_data[..., 1] * 4 + train_data[..., 2]
    pixels = pixels.flatten()
    counts = np.bincount(pixels, minlength=64)
    weights_np = np.zeros(65, dtype=np.float32)
    epsilon = 1e-6
    for i in range(1, 65):
        weights_np[i] = 1.0 / (counts[i-1] + epsilon)
    weights = torch.tensor(weights_np, dtype=torch.float32, device=device)
    print(f"[DEBUG] Class weights for tokens 0..64: {weights.cpu().numpy()}")
    criterion = nn.CrossEntropyLoss(weight=weights)
    
    trainer = Trainer(model, optimizer, scheduler, criterion,
                      train_loader, test_loader, device, vocab_size, seq_length)
    
    test_losses = [trainer.evaluate()]
    train_losses = []
    
    for epoch in range(num_epochs):
        print(f"=== Starting Epoch {epoch+1}/{num_epochs} ===")
        epoch_train_losses = trainer.train_epoch(epoch)
        train_losses.extend(epoch_train_losses)
        test_loss = trainer.evaluate()
        test_losses.append(test_loss)
        print(f"=== End of Epoch {epoch+1}: Test Loss = {test_loss:.4f} ===\n")
    
    samples = trainer.sample(100, image_shape, temperature=1.5)
    
    # Map tokens back to RGB:
    # Each token t in {1,...,64} is mapped as follows:
    #   R = (t-1) // 16, G = ((t-1) % 16) // 4, B = (t-1) % 4.
    def tokens_to_rgb(token_image):
        token_image = token_image.astype(np.int32) - 1
        R = token_image // 16
        G = (token_image % 16) // 4
        B = token_image % 4
        return np.stack([R, G, B], axis=-1).astype(np.uint8)
    
    samples_rgb = []
    for s in samples:
        token_image = s[..., 0]  # shape (H, W)
        img_rgb = tokens_to_rgb(token_image)
        samples_rgb.append(img_rgb)
    samples_rgb = np.stack(samples_rgb, axis=0)
    
    return np.array(train_losses), np.array(test_losses), samples_rgb

# Uncomment one of the following lines to run:
q3ab_save_results(1, 'b', q3_b)
# q3ab_save_results(2, 'b', q3_b)
