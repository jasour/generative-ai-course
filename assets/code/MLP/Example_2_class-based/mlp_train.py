import torch  # Import PyTorch
import torch.nn as nn  # Import PyTorch's neural network module (for defining layers)
import torch.optim as optim  # Import PyTorch's optimizer module
from mlp_model import MLP  # Import the MLP model from mlp_model.py
import torchvision  # Import torchvision for datasets and transformations
import torchvision.transforms as transforms  # Import transforms for preprocessing MNIST images
from torchsummary import summary  # Import torchsummary for visualizing the model structure

# Define Model Parameters (number of neurons in each layer)
model_params = {
    "input_size": 28 * 28,  # MNIST images are 28x28, so we flatten them into 784 features
    "hidden_size1": 128,  # Number of neurons in the first hidden layer
    "hidden_size2": 64,   # Number of neurons in the second hidden layer
    "output_size": 10      # Number of output classes (digits 0-9)
}

# Set device for computation (Use MPS for Mac, otherwise CPU)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Transformation pipeline that processes MNIST images before passing them to the model. 
transform = transforms.Compose([ # Applies multiple transformations sequentially
    transforms.ToTensor(),  # Convert images to PyTorch tensor (C*H*W format)
    transforms.Normalize((0.5,), (0.5,))  # Normalize images to [-1, 1] range, (Mean = 0.5, Std = 0.5),  (0.5,) makes it a tuple (0.5,0.5,0.5),
])

# Download and load the MNIST training dataset
train_dataset = torchvision.datasets.MNIST(
    root="./data",  # Directory where MNIST dataset (which consists of handwritten digit images (0-9)) will be stored
    train=True,  # Load training set
    transform=transform,  # Apply transformations
    download=True  # Download dataset if not available
)

# Create a DataLoader to load the dataset in batches
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,  # Use MNIST training dataset
    batch_size=64,  # Process images in batches of 64
    shuffle=True  # Shuffle data to improve training performance
)

# Initialize the model using the defined parameters
# **model_params unpacks the dictionary into function arguments
model = MLP(**model_params).to(device)  # Move the model to the selected device

# Print the model architecture
print(model)    

# Show detailed model summary (requires torchsummary)
device = torch.device("cpu")  # Use CPU for summary to avoid potential MPS issues
model.to(device)  # Move model to CPU
summary(model, (1, 28 * 28))  # Display model summary for a single input of size (1, 784)

# Define the loss function (CrossEntropyLoss for multi-class classification)
criterion = nn.CrossEntropyLoss()

# Define the optimizer (Adam optimizer for efficient training)
optimizer = optim.Adam(model.parameters(), lr=0.001)  # model.parameters() returns all trainable parameters of the model, Learning rate = 0.001

# Train the model for 5 epochs
num_epochs = 5
for epoch in range(num_epochs):  # Loop over each epoch
    # train_loader: A PyTorch DataLoader that loads the dataset in batches, 
    # enumerate(train_loader): Loops through the dataset and keeps track of the batch index (batch_idx).
    # (images, labels): A batch of images and their corresponding labels (targets), 
    for batch_idx, (images, labels) in enumerate(train_loader):  # Loop over each batch 

        images = images.view(-1, 28*28).to(device)  # Flatten images and move to device
        labels = labels.to(device)  # Move labels to device

        optimizer.zero_grad()  # Reset gradients before backward pass
        outputs = model(images)  # Forward pass: Compute predictions
        loss = criterion(outputs, labels)  # Compute loss between predictions and actual labels
        loss.backward()  # Backpropagation: Compute gradients
        optimizer.step()  # Update model weights using optimizer

        # Print loss every 200 batches
        if (batch_idx+1) % 200 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

# Save model architecture, parameters, and trained weights
checkpoint = {
    "model_state": model.state_dict(),  # Save trained model weights
    "model_params": model_params  # Save model architecture parameters
}
torch.save(checkpoint, "mlp_checkpoint.pth")  # Save to a file

# Print completion message
print(" Model training completed and saved as mlp_checkpoint.pth")