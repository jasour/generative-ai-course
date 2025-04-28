import torch  # PyTorch library for deep learning
import torch.nn as nn  # Module for building neural networks
import torch.optim as optim  # Optimizers for training models
import torch.nn.functional as F  # Contains activation functions and utilities
from torch.utils.data import DataLoader, Dataset  # For handling datasets and batching
import numpy as np  # NumPy for numerical operations
import dill  # ✅ Use dill instead of pickle
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer  # Use a pretrained tokenizer
import os

# ---------------- Configuration Class ----------------
class GPTConfig:
    """
    Stores the hyperparameters for the GPT model.
    """
    def __init__(self, vocab_size=50257, block_size=64, n_embd=128, n_head=4, n_layer=6, dropout=0.1, lr=3e-4):
        self.vocab_size = vocab_size  # Number of unique tokens in vocabulary
        self.block_size = block_size  # Maximum sequence length
        self.n_embd = n_embd  # Embedding size (dimension of token representations)
        self.n_head = n_head  # Number of attention heads in multi-head attention
        self.n_layer = n_layer  # Number of transformer blocks in the model
        self.dropout = dropout  # Dropout rate for regularization
        self.lr = lr  # Learning rate for the optimizer

# ---------------- Multi-Head Self-Attention ----------------
class MultiHeadSelfAttention(nn.Module):
    """
    Implements the Multi-Head Self-Attention mechanism.
    """
    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        assert n_embd % n_head == 0  # Ensure embedding size is divisible by number of heads
        self.n_head = n_head
        self.head_dim = n_embd // n_head  # Compute dimension per attention head
        self.scale = self.head_dim ** -0.5  # Scaling factor for attention scores

        # Linear transformations for Query, Key, and Value matrices
        self.qkv_proj = nn.Linear(n_embd, 3 * n_embd)  
        self.out_proj = nn.Linear(n_embd, n_embd)  # Linear projection for final output
        self.dropout = nn.Dropout(dropout)  # Dropout layer for regularization

    def forward(self, x, mask=None):
        """
        x: Input tensor of shape (batch_size, seq_length, embedding_dim)
        mask: Optional mask for padding or causal masking
        """
        B, T, C = x.shape  # B=batch_size, T=sequence_length, C=embedding_dim

        # 0) Compute Q, K, V matrices and reshape them for multi-head attention
        # 1) self.qkv_proj is a nn.Linear layer that projects the input tensor x into Query (Q), Key (K), and Value (V).
        # (Q,K,V) Shape: (B, T, 3 * n_embd)
        # 2) reshape: reshape it to separate Q, K, and V into three parts and split attention heads.
        # reshaped matrix: (B, T, 3, n_head, dhead_dim)
        # 3) permute: (3, B, n_head, T, head_dim), new shape: (3, B, n_head, T, head_dim)
        qkv = self.qkv_proj(x).reshape(B, T, 3, self.n_head, self.head_dim).permute(2, 0, 3, 1, 4)
        # K shape:  (B, n_head, T, head_dim), Q shape: (B, n_head, T, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Extract query, key, and value tensors

        # Compute attention scores (QK^T) and scale them
        # k.transpose(-2, -1): performs the transpose operation, but only on the last two dimensions (-2 and -1).
        attn_scores = (q @ k.transpose(-2, -1)) * self.scale  
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))  # Apply mask if provided

        # Compute attention weights by applying softmax to attn_scores
        attn_probs = F.softmax(attn_scores, dim=-1)  
        # Apply dropout, Randomly sets some values in attn_probs to zero with probability p.
        attn_probs = self.dropout(attn_probs)  

        # Compute attention output
        # attn_probs shape: (B, n_head, T, T)
        # v (Value Matrix) shape: (B, n_head, T, head_dim)
        # transpose: shape: (B, T, n_head, head_dim), want to merge the n_head and head_dim dimensions into a single embedding space.
        # reshape: flatten the last two dimensions into one, updated shape: (B, T, C = n_head * head_dim)
        attn_output = (attn_probs @ v).transpose(1, 2).reshape(B, T, C)
        return self.out_proj(attn_output)  # Final projection of attention output (B, T, C = n_head * head_dim)

# ---------------- Transformer Block ----------------
class TransformerBlock(nn.Module):
    """
    Implements a single transformer block consisting of:
    - Multi-head self-attention
    - Layer normalization
    - Feedforward neural network
    """
    def __init__(self, config):
        super().__init__()
        self.attn = MultiHeadSelfAttention(config.n_embd, config.n_head, config.dropout)  # Self-attention layer
        self.ln1 = nn.LayerNorm(config.n_embd)  # Layer normalization after attention
        self.ff = nn.Sequential(
            #  Expects input of shape (B, *, n_embd), Transforms it into shape (B, *, 4*n_embd).
            nn.Linear(config.n_embd, 4 * config.n_embd),  # Expand embedding dimension
            nn.GELU(),  # Non-linearity (GELU activation)
            nn.Linear(4 * config.n_embd, config.n_embd),  # Project back to original dimension
            nn.Dropout(config.dropout)  # Apply dropout, randomly sets some neurons to zero during training
        )
        self.ln2 = nn.LayerNorm(config.n_embd)  # Layer normalization after feedforward network

    def forward(self, x, mask=None):
        """
        Forward pass through the transformer block.
        """
        x = x + self.attn(self.ln1(x), mask)  # Apply self-attention and residual connection
        x = x + self.ff(self.ln2(x))  # Apply feedforward network and residual connection
        return x

# ---------------- GPT Model ----------------
class GPT(nn.Module):
    """
    Implements the full GPT model.
    """
    def __init__(self, config):
        super().__init__()
        self.token_emb = nn.Embedding(config.vocab_size, config.n_embd)  # Token embeddings
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))  # Positional embeddings
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])  # Transformer blocks
        self.ln_f = nn.LayerNorm(config.n_embd)  # Final layer normalization
        # Expects input of shape (B, *, n_embd), Transforms it into shape (B, *, Vocab_size).
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)  # Output layer

    def forward(self, x, mask=None):
        B, T = x.shape # Extracts the batch size (B) and sequence length (T) from x
        # Get token embeddings, Looks up the learned word embeddings for each token in x.
        #x: (B, T)  →  tok_emb: (B, T, config.n_embd)
        tok_emb = self.token_emb(x)  
        # Retrieves positional embeddings corresponding to the sequence length T.
        pos_emb = self.pos_emb[:, :T, :]  # Get positional embeddings
        x = tok_emb + pos_emb  # Combine token and position embeddings

        for block in self.blocks:  # Pass through each transformer block
            x = block(x, mask)

        x = self.ln_f(x)  # Apply final layer normalization
        logits = self.head(x)  # Compute logits for next-token prediction
        # Instead of explicitly applying softmax, PyTorch often uses nn.CrossEntropyLoss(), which applies softmax internally.
        return logits

# ---------------- Dummy Dataset ----------------
"""
class DummyDataset(Dataset):
    #Creates a random dataset for next-token prediction.
    def __init__(self, vocab_size, seq_len, dataset_size=1000):
        self.data = np.random.randint(0, vocab_size, (dataset_size, seq_len), dtype=np.int64)
        self.labels = np.roll(self.data, shift=-1, axis=1)  # Shifted labels for next-token prediction

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)
"""


class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer, block_size=32):
        """
        Loads a text file, tokenizes it, and prepares input-label pairs.
        """
        # ✅ Check if file exists and is not empty
        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            raise ValueError(f"Dataset file {file_path} is missing or empty!")

        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read().strip()  # ✅ Remove leading/trailing whitespace

        # ✅ Tokenize the text (Convert words into token IDs)
        tokens = tokenizer.encode(text)

        # ✅ Ensure there are enough tokens to create sequences
        if len(tokens) < block_size + 1:
            raise ValueError(f"Dataset has too few tokens ({len(tokens)}). Provide more text.")

        # ✅ Create input-label sequences
        self.inputs = []
        self.labels = []
        for i in range(len(tokens) - block_size):
            self.inputs.append(tokens[i : i + block_size])
            self.labels.append(tokens[i + 1 : i + block_size + 1])

        self.inputs = torch.tensor(self.inputs, dtype=torch.long)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

        print(f"✅ Loaded dataset: {len(self.inputs)} samples")  # ✅ Debugging output

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]

# ---------------- Training Function ----------------
def train(model, dataloader, optimizer, criterion, device, epochs=5):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            logits = model(inputs)
            
            # .view(-1, logits.size(-1)) reshapes logits: from (B, T, vocab_size) to ((B * T), vocab_size)
            # PyTorch’s nn.CrossEntropyLoss() expects inputs of shape: (N, C)  # N = number of samples, C = number of classes
            # N : Each token is treated as a separate sample, so we need (B * T).
            # C: Number of classes (vocab_size), which remains unchanged.
            
            # targets.view(-1): shape from (B, T) to (B * T,)
            # The batch and sequence dimensions are flattened into a single dimension.
            loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))  
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

# ---------------- Evaluation Function ----------------
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    print(f"Test Loss: {avg_loss:.4f}")


# ---------------- Save Model and Configuration ----------------
def save_model(model, config, model_path="gpt_model.pth", config_path="gpt_config.pkl"):
    torch.save(model.state_dict(), model_path)  # Save model weights
    with open(config_path, "wb") as f:
        dill.dump(config, f)  # ✅ Use dill to save the configuration
    print(f"Model saved to {model_path} and configuration to {config_path}")

# ---------------- Main Execution ----------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = GPTConfig()
    model = GPT(config).to(device)

    #dataset = DummyDataset(config.vocab_size, config.block_size, dataset_size=10000)
    # ✅ Load Pretrained GPT2 Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    # ✅ Load real text dataset
    dataset = TextDataset("data.txt", tokenizer, block_size=config.block_size)

    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    optimizer = optim.AdamW(model.parameters(), lr=config.lr)  
    criterion = nn.CrossEntropyLoss()

    train(model, train_loader, optimizer, criterion, device, epochs=50)

     # Save trained model and configuration
    save_model(model, config)