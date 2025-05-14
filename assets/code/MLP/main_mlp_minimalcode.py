
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt 





# ------------------------- Training Function -------------------------------------
def train(model, input_size, batch_size, num_epochs, learning_rate):
   
    # Define the dataset and dataloader
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]) # MNIST training set has 60,000 samples
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform) # len(train_dataset) → 60000. Each sample: (1, 28, 28) image and an integer label (0, 9)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True) # train_loader yields batches of data, each of the form: (images, labels). len(train_loader) → 937. Each batch: (batch_size, 1, 28, 28) images and (batch_size) labels.

    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        for batch_idx, (images, labels) in enumerate(train_loader): #batch_idx will go from 0 to (number of batches -1  = 937) 
            images = images.view(-1, input_size).to(device)  # Flatten and move to device, images: A tensor of shape [batch_size, 1, 28, 28]
            labels = labels.to(device)  # Move labels to device, labels: A tensor of shape [batch_size]
            outputs = model(images)
            loss = nn.CrossEntropyLoss()(outputs, labels)

            optimizer.zero_grad() # Zero the gradients before backward pass
            loss.backward() # Backpropagation: computes gradients of loss w.r.t. model parameters
            optimizer.step() # uses gradients to update weights

            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')

# ------------------------- Model Definition -------------------------------------
device = torch.device("mps") # or cpu
model_arch = [784, 128, 64, 10]
model = nn.Sequential(nn.Linear(784, 128),nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 10), nn.Identity() )
model.to(device)
# ------------------------- Model Train -----------------------------------------------
 # Pass the model into train()
train(model, input_size=model_arch[0], batch_size=64, num_epochs=5, learning_rate=0.001)




# ------------------------- Predict a label after training ----------------------------

# Load a single image from the test dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)

# Get one image and label
image, label = next(iter(test_loader))
image = image.view(-1, model_arch[0]).to(device)

# Predict using the trained model
with torch.no_grad():
    output = model(image)
    predicted_label = torch.argmax(output, dim=1).item()

print(f"Actual label: {label.item()} | Predicted label: {predicted_label}")
plt.imshow(image.cpu().view(28, 28), cmap="gray")
plt.show()