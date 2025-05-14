
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchinfo import summary


# ------------------------- MLP Model Definition ----------------------------
''' e.g., 
model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 10),
    nn.Identity()
)
'''
def mlp(sizes, activation, output_activation):
    layers = [] # e.g., sizes: [input layer, 1st hidden layer, 2nd hidden layer, output layer]  = [784 (flattened 28×28 image), 128, 64, 10]
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers) # unpacks the list layers into individual arguments nn.Sequential(layer1, activation1, layer2, activation2, ...).

# ------------------------- Training Function -------------------------------------
def train(model, input_size, batch_size, num_epochs, learning_rate):
  
    # Define the dataset and dataloader
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]) # MNIST training set has 60,000 samples
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform) # len(train_dataset) → 60000. Each sample: (1, 28, 28) image and an integer label (0, 9)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True) # train_loader yields batches of data, each of the form: (images, labels). len(train_loader) → 937. Each batch: (batch_size, 1, 28, 28) images and (batch_size) labels.

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        for batch_idx, (images, labels) in enumerate(train_loader): #batch_idx will go from 0 to (number of batches -1  = 937) 
            images = images.view(-1, input_size).to(device)  # Flatten and move to device, images: A tensor of shape [batch_size, 1, 28, 28]
            labels = labels.to(device)  # Move labels to device, labels: A tensor of shape [batch_size]
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad() # Zero the gradients before backward pass
            loss.backward() # Backpropagation: computes gradients of loss w.r.t. model parameters
            optimizer.step() # uses gradients to update weights

            if (batch_idx + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')
    # Print model summary
    print("Model Summary:")
    summary(model, input_size=(1, 784), device=device)
   

    # Save the model
    torch.save(model.state_dict(), 'mlp_model.pth')
    print("Model saved as 'mlp_model.pth'")





# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
input_size = 784
model_arch = [784, 128, 64, 10]
# ---------- Example Usage ----------
if __name__ == "__main__":

# Define model externally
    model = mlp(model_arch, activation=nn.ReLU, output_activation=nn.Identity)
    model.to(device)
 # Pass it into train()
    train(model, input_size=input_size, batch_size=64, num_epochs=5, learning_rate=0.001)

