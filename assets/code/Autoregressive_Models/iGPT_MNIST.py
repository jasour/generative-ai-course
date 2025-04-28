# https://github.com/rll/deepul/tree/master/homeworks/hw1

# Autoregressive Transformer on Shapes and MNIST
# Autoregressive Transformer to model binary MNIST and shapes images

# Download and load up the starter code.
# if [ -d deepul ]; then rm -Rf deepul; fi
# git clone https://github.com/rll/deepul.git
# pip install ./deepul
# pip install scikit-learn


"""
-iGPT uses learned positional encodings. We recommend to use those here as well. 
However, you may also use sinusoidal positional encodings if you wish (see the Attention is All You Need paper)
- Autoregressive transformer always predicts the next token, give prior tokens. 
iGPT has a special <bos> or beginning of sequence token at the start of every sequence every image. 
Make sure to include this in your implementation as well. You can generate unconditional sample by conditioning 
with the <bos> token.
- While dropout is a common feature in transformer models, you do not need to add it.
- Prebuilt transformers exist in some frameworks (i.e. pytorch). 
Don't just use an off the shelf implementation as the point of the exercise is to better understand the 
transformer architecture. Building the transformer from the ground up 
(use primitives such as Linear/Dense layers, LayerNorm, GeLU, Embedding)
- Learning rate warmup and cos learning rate decay are often used when training transformers to improve training 
stability and improve performance. See if this helps your model! Try 1000 steps of warmup with a cosine learning rate decay.


We recommend the following network design parameters:

-dmodel: 128
-heads: 4
-layers: 2
-GeLU nonlinearities

And the following hyperparameters:
-Batch size: 64 or 32 or 16 (whichever fits in your GPU)
-Learning rate: 10^-3
-15 epochs or more
-Adam Optimizer (this applies to all Transformers models trained in future parts)


Deliverables:
-Over the course of training, record the average negative log-likelihood (nats / dim) of the training data (per minibatch) and test data (for your entire test set). Code is provided that automatically plots the training curves.
-Report the final test set performance of your final model
-100 samples from the final trained model


def q3_a(train_data, test_data, image_shape, dset_id):
  
  train_data: A (n_train, H, W, 1) uint8 numpy array of color images with values in {0, 1}
  test_data: A (n_test, H, W, 1) uint8 numpy array of color images with values in {0, 1}
  image_shape: (H, W, 1), height, width, and # of channels of the image
  dset_id: An identifying number of which dataset is given (1 or 2). Most likely
           used to set different hyperparameters for different datasets

  Returns
  - a (# of training iterations,) numpy array of train_losses evaluated every minibatch
  - a (# of epochs + 1,) numpy array of test_losses evaluated once at initialization and after each epoch
  - a numpy array of size (100, H, W, 1) of samples with values in {0, 1}
  
  return train_losses, test_losses, samples

q3ab_save_results(1, 'a', q3_a)
q3ab_save_results(2, 'a', q3_a)

"""

import numpy as np
from deepul.hw1_helper import (
    # Q1
    visualize_q1_data,
    q1_sample_data_1,
    q1_sample_data_2,
    q1_save_results,
    # Q2
    q2a_save_results,
    q2b_save_results,
    visualize_q2a_data,
    visualize_q2b_data,
    # Q3
    q3ab_save_results,
    q3c_save_results,
    # Q4
    q4a_save_results,
    q4b_save_results,
    # Q5
    visualize_q5_data,
    q5a_save_results,
    # Q6
    visualize_q6_data,
    q6a_save_results,
)

"""
Below is one complete example implementation. In our code we build a transformer “from scratch” using PyTorch primitives 
(Linear layers, LayerNorm, GELU, and Embedding) and we treat each image as a 1D sequence with a prepended special 
beginning-of-sequence (bos) token. In our design the vocabulary is of size 3 (we reserve index 0 for the bos token and 
map the two binary pixel values 0 and 1 to indices 1 and 2 respectively). The transformer uses learned positional embeddings 
(one per sequence position) and a causal (lower-triangular) attention mask so that each output token is conditioned only on 
previous tokens. We also implement a learning-rate scheduler that performs 1000 warmup steps followed by cosine decay. 
Finally, during training we record the average negative log-likelihood (in nats per image dimension) per minibatch 
(for training) and per epoch (for testing) and generate 100 unconditional samples from the final model.
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
def debug_visualize_images(data, num_images=16):
    """
    Visualize a random subset of images from data, which should have shape (n, H, W, 1).
    """
    indices = np.random.choice(len(data), size=num_images, replace=False)
    sample = data[indices]
    grid_size = int(np.ceil(np.sqrt(num_images)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(2 * grid_size, 2 * grid_size))
    axes = axes.flatten()
    for i in range(num_images):
        axes[i].imshow(sample[i].squeeze(), cmap='gray')
        axes[i].axis('off')
    for i in range(num_images, len(axes)):
        axes[i].axis('off')
    plt.suptitle("Debug Visualization of Training Data")
    plt.show()

# -------------------------
# Data Processing Components
# -------------------------
class ImageSequenceDataset(Dataset):
    """
    Transforms each binary image into a sequence.
    The pixel values (0,1) are mapped to (1,2) while index 0 is reserved for the <bos> token.
    """
    def __init__(self, data):
        # data: numpy array of shape (n, H, W, 1) with values in {0,1}
        self.data = data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        img = self.data[idx].squeeze(-1)   # shape (H, W)
        img = img.flatten()                # shape (H*W,)
        img = img.astype(np.int64) + 1     # map: 0 -> 1, 1 -> 2
        # Prepend the <bos> token (0)
        seq = np.concatenate(([0], img))
        return torch.tensor(seq, dtype=torch.long)

# -------------------------
# Model Components
# -------------------------
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
    
    def forward(self, x, mask):
        B, T, C = x.size()
        qkv = self.qkv(x)  # (B, T, 3*d_model)
        qkv = qkv.reshape(B, T, self.n_heads, 3 * self.d_head)  # (B, T, n_heads, 3*d_head)
        qkv = qkv.permute(0, 2, 1, 3)                            # (B, n_heads, T, 3*d_head)
        q, k, v = torch.chunk(qkv, 3, dim=-1)  # each (B, n_heads, T, d_head)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)  # (B, n_heads, T, T)
        attn_scores = attn_scores.masked_fill(~mask, float('-inf'))
        attn = torch.softmax(attn_scores, dim=-1)
        out = torch.matmul(attn, v)  # (B, n_heads, T, d_head)
        out = out.transpose(1, 2).reshape(B, T, C)
        out = self.out_proj(out)
        return out

class FeedForward(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.fc1 = nn.Linear(d_model, 4 * d_model)
        self.fc2 = nn.Linear(4 * d_model, d_model)
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
        token_emb = self.token_embedding(x)              # (B, T, d_model)
        pos_emb = self.pos_embedding[:T].unsqueeze(0)      # (1, T, d_model)
        x = token_emb + pos_emb
        # Create causal mask: allow each position to attend only to itself and previous tokens.
        mask = torch.tril(torch.ones(T, T, device=x.device)).bool()
        for layer in self.layers:
            x = layer(x, mask)
        x = self.ln_final(x)
        logits = self.head(x)  # (B, T, vocab_size)
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
            loss = self.criterion(logits.reshape(-1, self.vocab_size),
                                  targets.reshape(-1))
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
                loss = self.criterion(logits.reshape(-1, self.vocab_size),
                                      targets.reshape(-1))
                total_loss += loss.item() * targets.numel()
                total_tokens += targets.numel()
        return total_loss / total_tokens

    def sample(self, num_samples, image_shape, temperature=1.0):
        """
        Generates images by starting from <bos> (token=0) and sampling one token at a time.
        Includes temperature scaling for more diverse sampling.
        """
        self.model.eval()
        samples = []
        with torch.no_grad():
            for sidx in range(num_samples):
                generated = torch.tensor([[0]], dtype=torch.long, device=self.device)  # shape (1,1)
                for _ in range(self.seq_length - 1):
                    logits = self.model(generated)    # (1, current_length, vocab_size)
                    logits = logits[:, -1, :]         # (1, vocab_size) for the last token
                    # Mask out bos token (index 0) so it is never sampled beyond position 0:
                    logits[:, 0] = -float('inf')
                    # Apply temperature scaling
                    probs = torch.softmax(logits / temperature, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)  # (1,1)
                    generated = torch.cat((generated, next_token), dim=1)
                # Debug: Print generated token sequence (before mapping) for first few samples
                if sidx < 3:
                    tokens = generated.squeeze().cpu().numpy()
                    print(f"[Sampling Debug] Sample {sidx} - Raw token sequence: {tokens}")
                    print(f"[Sampling Debug] Unique tokens before mapping: {np.unique(tokens)}")
                # Remove the <bos> token and map tokens: 1 -> 0, 2 -> 1
                img_tokens = generated[0, 1:] - 1
                H, W, C = image_shape
                img = img_tokens.reshape(H, W, C).cpu().numpy().astype(np.uint8)
                samples.append(img)
                if sidx < 3:
                    print(f"[Sampling Debug] Sample {sidx} - Unique pixel values after mapping: {np.unique(img)}")
        return np.stack(samples, axis=0)

# -------------------------
# Main Function: q3_a
# -------------------------
def q3_a(train_data, test_data, image_shape, dset_id):
    """
    Trains an autoregressive transformer to model binary MNIST or shapes images.
    
    Args:
      train_data: (n_train, H, W, 1) uint8 numpy array with values in {0, 1}
      test_data: (n_test, H, W, 1) uint8 numpy array with values in {0, 1}
      image_shape: (H, W, 1) tuple giving image dimensions.
      dset_id: Identifying dataset (1 or 2) to possibly set different hyperparameters.
    
    Returns:
      train_losses: numpy array of training losses (nats/dim) recorded every minibatch.
      test_losses: numpy array of test losses (evaluated at initialization and after each epoch).
      samples: numpy array of size (100, H, W, 1) with values in {0, 1}
    """
    # Use MPS if available, otherwise CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    H, W, C = image_shape
    seq_length = 1 + H * W
    vocab_size = 3  # 0: <bos>, 1: pixel 0, 2: pixel 1

    # Debug checks on input data
    print("[DEBUG] train_data shape:", train_data.shape)
    print("[DEBUG] unique pixel values in train_data:", np.unique(train_data))
    print("[DEBUG] test_data shape:", test_data.shape)
    print("[DEBUG] unique pixel values in test_data:", np.unique(test_data))
    debug_visualize_images(train_data, num_images=16)

    # Create datasets and dataloaders
    train_dataset = ImageSequenceDataset(train_data)
    test_dataset = ImageSequenceDataset(test_data)
    batch_size = 64
    num_epochs = 15
    lr = 1e-3
    warmup_steps = 1000

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Instantiate the model
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

    # Compute class weights from training data (excluding the bos token)
    # Pixels in train_data are 0 or 1; they are mapped as:
    #  pixel 0 -> token 1 and pixel 1 -> token 2.
    pixels = train_data[..., 0].flatten()
    count_token1 = np.sum(pixels == 0)  # number of pixel 0's
    count_token2 = np.sum(pixels == 1)  # number of pixel 1's
    w1 = 1.0 / (count_token1 + 1e-6)
    w2 = 1.0 / (count_token2 + 1e-6)
    # Create weights for all 3 tokens: bos token (0) gets weight 0.
    weights = torch.tensor([0.0, w1, w2], dtype=torch.float32, device=device)
    print(f"[DEBUG] Class weights for tokens 0, 1 and 2: {weights.cpu().numpy()}")
    criterion = nn.CrossEntropyLoss(weight=weights)

    # Create trainer
    trainer = Trainer(model, optimizer, scheduler, criterion,
                      train_loader, test_loader, device, vocab_size, seq_length)

    # Initial evaluation
    test_losses = [trainer.evaluate()]
    train_losses = []

    for epoch in range(num_epochs):
        print(f"=== Starting Epoch {epoch+1}/{num_epochs} ===")
        epoch_train_losses = trainer.train_epoch(epoch)
        train_losses.extend(epoch_train_losses)
        test_loss = trainer.evaluate()
        test_losses.append(test_loss)
        print(f"=== End of Epoch {epoch+1}: Test Loss = {test_loss:.4f} ===\n")

    # Generate 100 samples using a specified temperature (try experimenting, e.g. temperature=1.5 or higher)
    samples = trainer.sample(100, image_shape, temperature=1.5)

    return np.array(train_losses), np.array(test_losses), samples

# Example usage (uncomment one of these lines to run):
#q3ab_save_results(1, 'a', q3_a)
q3ab_save_results(2, 'a', q3_a)