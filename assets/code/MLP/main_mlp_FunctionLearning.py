import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


# ------------------------- Training Function -------------------------------------
def train(model, batch_size, num_epochs, learning_rate):
   
    # Generate 2D Sample Dataset
    num_samples=5000
    x = torch.rand(num_samples, 2) * 4 * torch.pi - 2 * torch.pi  # Range: [-2π, 2π]
    y = torch.sin(x[:, 0]) + torch.cos(x[:, 1])
    dataset = torch.utils.data.TensorDataset(x, y.unsqueeze(1) ) # This creates a PyTorch-compatible dataset object.  dataset[i] → (input_tensor, label_tensor) # y.unsqueeze(1): Make y shape [N, 1] required for PyTorch loss functions
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True) # train_loader yields batches of data, each of the form: (inputs, labels). len(train_loader) → 78. Each batch: (batch_size, 2) inputs and (batch_size, 1) labels.

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
 
    for epoch in range(num_epochs):
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = nn.MSELoss()(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch + 1}/{num_epochs}] Loss: {loss.item():.4f}")


# ------------------------- Model Definition -------------------------------------
device = torch.device("mps")
model = nn.Sequential( nn.Linear(2, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))
model.to(device)

# ------------------------- Train the Model --------------------------------------
train(model, batch_size=64, num_epochs=100, learning_rate=0.001)

# ------------------------- Test Prediction & Plot -------------------------------
# Create a meshgrid to visualize the function
x1 = torch.linspace(-2 * torch.pi, 2 * torch.pi, 100)
x2 = torch.linspace(-2 * torch.pi, 2 * torch.pi, 100)
xx1, xx2 = torch.meshgrid(x1, x2, indexing="ij")
grid = torch.stack([xx1.flatten(), xx2.flatten()], dim=1).to(device)

with torch.no_grad():
    preds = model(grid).view(100, 100).cpu()

plt.figure(figsize=(6, 5))
plt.contourf(xx1, xx2, preds, levels=50, cmap="viridis")
plt.colorbar(label="Predicted y")
plt.title("Learned Approximation of y = sin(x1) + cos(x2)")
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()

