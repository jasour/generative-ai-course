# ------------------------- ViT-style Transformer for MNIST -------------------------
'''
Patch-based tokenization and Transformer encoder for MNIST digit classification.
 ViT-style transformer to classify 28×28 grayscale images (MNIST) by:
	•	Splitting the image into small patches
	•	Turning each patch into a token
	•	Feeding these tokens into a Transformer encoder
	•	Using a [CLS] token to summarize the image and predict the digit class

1. Patch Embedding
	•	Each image is divided into 7×7 non-overlapping patches → 16 total
	•	Each patch is flattened to a vector of size 49
	•	This becomes your sequence of image tokens

2. Input Projection
	•	Each 49-dimensional patch vector is projected to d_model = 64 dimensional embedding vector
	•	Now, the input becomes a sequence of 16 tokens of size 64

3. [CLS] Token + Positional Encoding
	•	A learnable [CLS] token is prepended to the sequence (now 17 tokens)
	•	Positional encoding ((17, 64)) is added to preserve spatial order of patches

4. Transformer Encoder
	•	Each token (including CLS) attends to all others
	•	The attention mechanism lets the model learn relationships between patches
	•	After num_layers transformer blocks, each token has been updated contextually

5. Classification
	•	Only the CLS token’s final embedding is passed to an MLP head
	•	The MLP outputs 10 logits → one per digit class (0–9)

'''

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import math

# ---------------- Positional Encoding ---------------------------------------
def positional_encoding(max_len, d_model, device):
    # Create a (max_len, d_model) tensor to hold positional encodings.  
    # max_len = num_patches + 1, num_patches: Number of image patches (e.g., 28×28 image with 7×7 patches = 16 patches),
    # +1: We add 1 extra position for the [CLS] token, which is used to summarize the image for classification
    pe = torch.zeros(max_len, d_model, device=device)
    # Positions: [0, 1, ..., max_len-1] as a column
    position = torch.arange(0, max_len, device=device).unsqueeze(1)
    # Compute frequency terms for sine/cosine functions
    div_term = torch.exp(torch.arange(0, d_model, 2, device=device) * (-math.log(10000.0) / d_model))
    # Apply sine to even indices
    pe[:, 0::2] = torch.sin(position * div_term)
    # Apply cosine to odd indices
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

# ---------------- Transformer Encoder Builder -------------------------------
def build_transformer_encoder(d_model, nhead, num_layers, dim_feedforward): 
    layers = []
    for _ in range(num_layers):
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward)
        layers.append(layer)
    return nn.Sequential(*layers)

# ---------------- Build Vision Transformer Model ----------------------------
def build_model(num_patches, patch_dim, num_classes, d_model, nhead, num_layers, dim_feedforward, device):
    # Linear layer to project each patch to d_model dimensions
    input_proj = nn.Linear(patch_dim, d_model).to(device)
    # Build transformer encoder
    transformer = build_transformer_encoder(d_model, nhead, num_layers, dim_feedforward).to(device)
    # Positional encoding for [CLS] + all patches
    pos_enc = positional_encoding(num_patches + 1, d_model, device=device)
    # Learnable [CLS] token
    cls_token = nn.Parameter(torch.zeros(1, 1, d_model, device=device))
    # Output MLP head: Normalize and project CLS token to num_classes
    output_proj = nn.Sequential(
        nn.LayerNorm(d_model),
        nn.Linear(d_model, num_classes)
    ).to(device)
    return input_proj, transformer, pos_enc, cls_token, output_proj

# ---------------- Transformer Forward Pass ----------------------------------
def transformer_forward(x, input_proj, transformer, pos_enc, cls_token, output_proj, patch_size):
    B, C, H, W = x.shape  # Input shape: (B, 1, 28, 28)
    P = patch_size  # Patch size (e.g., 7)
    # Split image into non-overlapping patches using unfold
    x = x.unfold(2, P, P).unfold(3, P, P)  # → (B, C, H/P, W/P, P, P)
    x = x.contiguous().view(B, C, -1, P * P).permute(0, 2, 3, 1).reshape(B, -1, P * P)  # → (B, N, patch_dim)
    # Project patches to d_model dimensions
    x = input_proj(x)  # → (B, N, d_model)
    # Expand and prepend CLS token
    cls_tok = cls_token.expand(-1, B, -1)  # → (1, B, d_model)
    x = torch.cat([cls_tok, x.permute(1, 0, 2)], dim=0)  # → (1+N, B, d_model)
    # Add positional encodings
    x = x + pos_enc[:x.size(0)].unsqueeze(1)
    # Pass through transformer layers
    for layer in transformer:
        x = layer(x)
    # Return classification output based on CLS token
    return output_proj(x[0])  # → (B, num_classes)

# ---------------- Training and Evaluation -----------------------------------
def train_and_eval(device):
    # Model hyperparameters
    img_size = 28
    patch_size = 7
    num_classes = 10
    d_model = 64
    nhead = 4
    num_layers = 2
    dim_feedforward = 128
    patch_dim = patch_size * patch_size
    num_patches = (img_size // patch_size) ** 2

    # Training hyperparameters
    batch_size = 128
    num_epochs = 5
    lr = 1e-3

    # Load MNIST dataset
    transform = transforms.ToTensor()
    train_set = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    test_set = datasets.MNIST(root="./data", train=False, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    # Initialize model parts
    input_proj, transformer, pos_enc, cls_token, output_proj = build_model(
        num_patches, patch_dim, num_classes, d_model, nhead, num_layers, dim_feedforward, device)

    # Collect parameters for optimizer
    params = list(input_proj.parameters()) + list(output_proj.parameters()) + [cls_token]
    for layer in transformer:
        params += list(layer.parameters())

    optimizer = optim.Adam(params, lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(num_epochs):
        input_proj.train(), transformer.train(), output_proj.train()
        total_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = transformer_forward(xb, input_proj, transformer, pos_enc, cls_token, output_proj, patch_size)
            loss = loss_fn(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader.dataset):.4f}")

    # Evaluation loop
    input_proj.eval(), transformer.eval(), output_proj.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = transformer_forward(xb, input_proj, transformer, pos_enc, cls_token, output_proj, patch_size)
            preds = logits.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
    print(f"Test Accuracy: {100. * correct / total:.2f}%")
    return input_proj, transformer, pos_enc, cls_token, output_proj, patch_size

# ---------------- Predict One Sample ---------------------------------------
def predict_sample(model_parts, device):
    # Load single image from test set
    transform = transforms.ToTensor()
    mnist = datasets.MNIST(root="./data", train=False, transform=transform)
    image, label = mnist[0]

    # Show image
    plt.imshow(image.squeeze(), cmap="gray")
    plt.title(f"True Label: {label}")
    plt.axis("off")
    plt.show()

    # Prepare and run inference
    image = image.unsqueeze(0).to(device)  # → (1, 1, 28, 28)
    input_proj, transformer, pos_enc, cls_token, output_proj, patch_size = model_parts
    input_proj.eval(), transformer.eval(), output_proj.eval()
    with torch.no_grad():
        logits = transformer_forward(image, input_proj, transformer, pos_enc, cls_token, output_proj, patch_size)
        pred = logits.argmax(dim=1).item()
    print(f"Predicted Label: {pred}")

# ---------------- Run Script ------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
model_parts = train_and_eval(device)
predict_sample(model_parts, device)
