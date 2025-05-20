# ------------------------- Transformer for Time Series Forecasting -------------------------
# A simplified Transformer-based model for univariate time series forecasting.
# Goal: Predict the next value in a sequence based on a fixed-length history using self-attention.

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import math

# ---------------- Positional Encoding ---------------------------------------
# Positional encoding for time series input. Each time step gets a unique vector of shape (d_model,).
# Input: max_len: number of time steps, d_model: embedding dimension
# Output: Tensor of shape (max_len, d_model) for encoding sequence positions

def positional_encoding(max_len, d_model, device): 
    pe = torch.zeros(max_len, d_model, device=device)
    position = torch.arange(0, max_len, device=device).unsqueeze(1)  # column vector [0, 1, ..., max_len-1]
    div_term = torch.exp(torch.arange(0, d_model, 2, device=device) * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)  # even indices
    pe[:, 1::2] = torch.cos(position * div_term)  # odd indices
    return pe

# ---------------- Build Transformer Encoder (Stack of Blocks) ---------------
# Each block includes multi-head attention, feedforward layers, and normalization.

def build_transformer_encoder(d_model, nhead, num_layers, dim_feedforward): 
    layers = []
    for _ in range(num_layers):
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward)
        layers.append(layer)
    return nn.Sequential(*layers)

# ---------------- Build Model Parts -----------------------------------------
# Model structure: input projection -> positional encoding -> transformer encoder -> MLP output

def build_model(seq_len, d_model=32, nhead=2, num_layers=2, dim_feedforward=128, device="mps"):
    input_proj = nn.Linear(1, d_model).to(device)  # Project scalar to d_model-dim vector
    transformer = build_transformer_encoder(d_model, nhead, num_layers, dim_feedforward).to(device)
    pos_enc = positional_encoding(seq_len, d_model, device=device)  # Shape: (seq_len, d_model)
    output_proj = nn.Sequential(
        nn.Linear(d_model, 32),
        nn.ReLU(),
        nn.Linear(32, 1)
    ).to(device)  # Final MLP to predict next value
    return input_proj, transformer, pos_enc, output_proj

# ---------------- Forward Pass Function -------------------------------------
# Input: x shape (B, T), where T is the sequence length
# Returns: prediction for next time step (B, 1)

def transformer_forward(x, input_proj, transformer, pos_enc, output_proj):
    x = x.unsqueeze(-1).permute(1, 0, 2)          # Reshape to (T, B, 1)
    x = input_proj(x)                             # (T, B, d_model)
    x = x + pos_enc.unsqueeze(1)                  # Add positional encoding (T, 1, d_model)

    # Create causal mask to prevent attention to future positions
    seq_len = x.size(0)
    attn_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()

    for layer in transformer:
        x = layer(x, src_mask=attn_mask)          # Apply attention mask

    return output_proj(x[-1])                     # Output for the last time step (B, d_model) -> (B, 1)

# ---------------- Training Loop ---------------------------------------------
# Uses sine wave data with noise. Trains model to predict next time step.

def train(model_parts, data, seq_len, device, epochs=100, batch_size=64):
    input_proj, transformer, pos_enc, output_proj = model_parts
    x_seq, y_seq = [], []
    for i in range(len(data) - seq_len):
        x_seq.append(data[i:i+seq_len])
        y_seq.append(data[i+seq_len])
    x_seq = torch.stack(x_seq)
    y_seq = torch.stack(y_seq).unsqueeze(1)

    dataset = torch.utils.data.TensorDataset(x_seq, y_seq)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    params = list(input_proj.parameters()) + list(output_proj.parameters())
    for layer in transformer:
        params += list(layer.parameters())

    optimizer = optim.Adam(params, lr=0.001)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = transformer_forward(xb, input_proj, transformer, pos_enc, output_proj)
            loss = loss_fn(pred, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# ---------------- Forecast Function ------------------------------------------
# Autoregressively predicts `steps` future values from a seed sequence

def predict(model_parts, seed_seq, steps, device):
    input_proj, transformer, pos_enc, output_proj = model_parts
    pred_seq = seed_seq.clone().to(device)
    preds = []
    for _ in range(steps):
        with torch.no_grad():
            y = transformer_forward(pred_seq.unsqueeze(0), input_proj, transformer, pos_enc, output_proj)
        preds.append(y.item())
        pred_seq = torch.cat([pred_seq[1:], y.squeeze(0)])
    return preds

# ---------------- Main Execution --------------------------------------------
device = torch.device("mps" if torch.cuda.is_available() else "cpu")
seq_len = 20

# Generate noisy sine wave as training data
time = torch.linspace(0, 100, 2000)
signal = torch.sin(time) + 0.1 * torch.randn_like(time)

# Build and train model
model = build_model(seq_len=seq_len, device=device)
train(model, signal, seq_len, device)

# Forecast 100 future steps using the last training window as seed
seed = signal[:seq_len].clone()
future = predict(model, seed, steps=100, device=device)

# Plot seed, forecast, and training data
plt.plot(range(len(signal)), signal.cpu(), label="Training Data", alpha=0.3, color="lightblue")
plt.plot(range(seq_len), seed.cpu(), label="Seed", color="orange")
plt.plot(range(seq_len, seq_len + 100), future, label="Forecast", color="green")
plt.title("Time Series Forecast with Transformer")
plt.xlabel("Time Step")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.show()
