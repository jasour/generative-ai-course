import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import math


# ------------------- Positional Encoding for 2D Input -----------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        self.pe = pe.unsqueeze(1)  # Shape: (max_len, 1, d_model)

    def forward(self, x):
        return x + self.pe.to(x.device)


# ------------------------- Transformer Model ----------------------------------
class TransformerFunctionApproximator(nn.Module):
    def __init__(self, d_model=32, nhead=4, num_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=64)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_proj = nn.Sequential(
            nn.Linear(d_model * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # x: (B, 2) → (2, B, 1)
        x = x.unsqueeze(-1).permute(1, 0, 2)
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2).reshape(x.size(1), -1)  # Flatten sequence: (B, 2*d_model)
        return self.output_proj(x)


# ------------------------- Training Function -------------------------------------
def train(model, batch_size, num_epochs, learning_rate):
    num_samples = 5000
    x = torch.rand(num_samples, 2) * 4 * torch.pi - 2 * torch.pi
    y = torch.sin(x[:, 0]) + torch.cos(x[:, 1])
    dataset = torch.utils.data.TensorDataset(x, y.unsqueeze(1))
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = nn.MSELoss()(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {loss.item():.4f}")


# ------------------------- Device & Model ---------------------------------------
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
model = TransformerFunctionApproximator().to(device)

# ------------------------- Train the Model --------------------------------------
train(model, batch_size=64, num_epochs=100, learning_rate=0.001)

# ------------------------- Test Prediction & Plot -------------------------------
x1 = torch.linspace(-2 * torch.pi, 2 * torch.pi, 100)
x2 = torch.linspace(-2 * torch.pi, 2 * torch.pi, 100)
xx1, xx2 = torch.meshgrid(x1, x2, indexing="ij")
grid = torch.stack([xx1.flatten(), xx2.flatten()], dim=1).to(device)

with torch.no_grad():
    preds = model(grid).view(100, 100).cpu()

plt.figure(figsize=(6, 5))
plt.contourf(xx1, xx2, preds, levels=50, cmap="viridis")
plt.colorbar(label="Predicted y")
plt.title("Transformer Approximation of y = sin(x1) + cos(x2)")
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()