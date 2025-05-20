import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import math


'''
2D function approximation  ( f(x1,x2) ) using a Transformer model: 
In this Transformer-based model, where the input is a 2D point [x1, x2] treated as a sequence of two scalars, the self-attention mechanism is applied between x1 and x2.

1. Input:
It reshape the input [x1, x2] of shape (B, 2) into a sequence of 2 tokens per example.  x = [ x_1 ,  x_2 ]  from shape (B, 2) → shape: (2, B, 1)

2.	Projection + Positional Encoding:
Each scalar is projected to a d_model-dimensional vector, and positional encodings are added:  token_1 = Proj(x_1) + pos_0 ,  token_2 = Proj(x_2) + pos_1

3.	Self-Attention Layer:
The Transformer encoder  (Transfomer Block) now applies self-attention over the 2-token sequence: 	•	x1 attends to x2	•	x2 attends to x1	•	Both are updated based on each other
'''

# ---------------- Positional Encoding ---------------------------------------
#  Positional encoding for 2D input, max_len: 2, d_model: 32
#  Tensor of shape: (max_len, d_model)
# e.g., output tensor for d_model=4 : ([  [sin(0), cos(0), sin(0), cos(0)],
#                                         [sin(1*w1), cos(1*w1), sin(1*w2), cos(1*w2)]  ]) where w1, w2 are the frequencies.
def positional_encoding(max_len, d_model, device): 
    pe = torch.zeros(max_len, d_model, device=device)
    position = torch.arange(0, max_len, device=device).unsqueeze(1) # constructing a column vector of integer positions, which represent the index of each token (or input element) in the sequence. e.g. max_len=2: tensor([[0],[1]], device='mps:0')
    div_term = torch.exp(torch.arange(0, d_model, 2, device=device) * (-math.log(10000.0) / d_model)) # creates a vector of frequency scaling terms. e.g., d_model =4, tensor([1.0000, 0.0100], device='mps:0')
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe  # Tensor of shape: (max_len, d_model), e.g., tensor([   [0.0000, 1.0000, 0.0000, 1.0000] ,  [0.8415, 0.5403, 0.0100, 1.0000]  ], device='mps:0')


# ---------------- Create the Transformer Encoder (A single Transformer block) ----------------------------
# num_layers: number of transformer blocks, each block has multiple attention heads, feedforward layers, and layer normalization. and residual connections
# dim_feedforward: hidden layer size of the feedforward. It consists of two linear transformations with a ReLU activation in between, e.g.,
                 # Input:shape (B, d_model = 32), Layer 1: Linear(32 → dim_feedforward=128),  ReLU, Layer 2: Linear(128 → 32), Output:  shape (B, 32)
def build_transformer_encoder(d_model, nhead, num_layers, dim_feedforward): 
    layers = []
    for _ in range(num_layers):
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward)
        layers.append(layer)
    return nn.Sequential(*layers)


# ---------------- Setup Model Parts -----------------------------------------
def build_model(d_model=32, nhead=4, num_layers=2, dim_feedforward=64, device="cpu"):
    input_proj = nn.Linear(1, d_model).to(device) # Projects a scalar input (1 value) into a d_model-dimensional vector.
    transformer = build_transformer_encoder(d_model, nhead, num_layers, dim_feedforward)
    transformer.to(device)
    pos_enc = positional_encoding(2, d_model, device=device)
    output_proj = nn.Sequential(   # After the Transformer processes two vectors of size d_model, we flatten them into a single vector of size 2 * d_modelf followed by a MLP
        nn.Linear(2 * d_model, 64),
        nn.ReLU(),
        nn.Linear(64, 1)
    ).to(device)
    return input_proj, transformer, pos_enc, output_proj


# ---------------- Forward Pass Function -------------------------------------
def transformer_forward(x, input_proj, transformer, pos_enc, output_proj):
    # x: (B, 2) → reshape to sequence: (2, B, 1)
    x = x.unsqueeze(-1).permute(1, 0, 2)                  # → (seq_len=2, batch_size, 1)
    x = input_proj(x)                                    # → (2, B, d_model)
    x = x + pos_enc.unsqueeze(1)                         # Add positional encoding → (2, B, d_model)
    for layer in transformer:
        x = layer(x)                                     # Transformer layers
    x = x.permute(1, 0, 2).reshape(x.size(1), -1)         # (2, B, d_model) → (B, 2*d_model)
    return output_proj(x)                                # → (B, 1)


# ---------------- Training Loop ---------------------------------------------
def train(model_parts, batch_size, num_epochs, learning_rate, device):
    input_proj, transformer, pos_enc, output_proj = model_parts

    # Generate data
    num_samples = 5000
    x = torch.rand(num_samples, 2) * 4 * torch.pi - 2 * torch.pi
    y = torch.sin(x[:, 0]) + torch.cos(x[:, 1])
    dataset = torch.utils.data.TensorDataset(x, y.unsqueeze(1))
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    params = list(input_proj.parameters()) + list(output_proj.parameters())
    for layer in transformer:
        params += list(layer.parameters())

    optimizer = optim.Adam(params, lr=learning_rate)

    for epoch in range(num_epochs):
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = transformer_forward(inputs, input_proj, transformer, pos_enc, output_proj)
            loss = nn.MSELoss()(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {loss.item():.4f}")





# ---------------- Inference and Plotting ------------------------------------
def predict_and_plot(model_parts, device):
    input_proj, transformer, pos_enc, output_proj = model_parts

    x1 = torch.linspace(-2 * torch.pi, 2 * torch.pi, 100)
    x2 = torch.linspace(-2 * torch.pi, 2 * torch.pi, 100)
    xx1, xx2 = torch.meshgrid(x1, x2, indexing="ij")
    grid = torch.stack([xx1.flatten(), xx2.flatten()], dim=1).to(device)

    with torch.no_grad():
        preds = transformer_forward(grid, input_proj, transformer, pos_enc, output_proj).view(100, 100).cpu()

    plt.figure(figsize=(6, 5))
    plt.contourf(xx1, xx2, preds, levels=50, cmap="viridis")
    plt.colorbar(label="Predicted y")
    plt.title("Transformer Approximation of y = sin(x1) + cos(x2)")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.show()


# ---------------- Main Execution --------------------------------------------
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
model = build_model(device=device)
print(model)
train(model, batch_size=64, num_epochs=50, learning_rate=0.001, device=device)
predict_and_plot(model, device=device)