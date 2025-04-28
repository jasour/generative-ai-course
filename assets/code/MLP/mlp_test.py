import torch  # Import PyTorch for deep learning
import torchvision  # Import torchvision for dataset handling
import torchvision.transforms as transforms  # Import transforms for preprocessing MNIST images
import torch.nn.functional as F  # Import functional operations (for Softmax)
import matplotlib.pyplot as plt  # Import Matplotlib for visualization
from mlp_model import MLP  # Import the MLP model

# Set device for computation (Use MPS for Mac, otherwise CPU)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Load the trained model checkpoint
checkpoint = torch.load("mlp_checkpoint.pth", map_location=device)

# Retrieve model parameters and initialize model
model_params = checkpoint["model_params"]  # Load model architecture parameters
model = MLP(**model_params).to(device)  # Create model using saved parameters

# Load trained model weights
model.load_state_dict(checkpoint["model_state"])
model.eval()  # Set model to evaluation mode (disables dropout, batch normalization, etc.)

# Define MNIST transformation (must match the transformations used during training)
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert image to PyTorch tensor (C, H, W format)
    transforms.Normalize((0.5,), (0.5,))  # Normalize pixel values to range [-1, 1]
])

# Load the MNIST test dataset
test_dataset = torchvision.datasets.MNIST(
    root="./data",  # Directory where dataset is stored
    train=False,  # Load test set (not training set)
    transform=transform,  # Apply transformations
    download=True  # Download if not available
)

# Create a DataLoader to iterate through the dataset
test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset, 
    batch_size=1,  # Load one image at a time for testing
    shuffle=True  # Randomize test images
)

# Get a single test image and its corresponding label
image, label = next(iter(test_loader))

# Flatten the image to match the input format expected by MLP (1D vector of 28x28 = 784)
image = image.view(-1, 28*28).to(device)

# Perform inference (make a prediction)
output = model(image)  # Get raw output logits
probs = F.softmax(output, dim=1)  # Apply Softmax to get probabilities

# Get the predicted class (digit with the highest probability)
predicted = torch.argmax(probs, dim=1).item()

# Get the confidence score of the predicted digit
confidence = probs.max().item()

# Print the actual and predicted labels with confidence score
print(f"Actual: {label.item()} | Predicted: {predicted} (Confidence: {confidence:.4f})")

# Print probabilities for all digit classes (0-9)
for i, prob in enumerate(probs.squeeze().tolist()):
    print(f"Digit {i}: {prob:.4f}")

# Display the test image
plt.imshow(image.cpu().view(28, 28), cmap="gray")
plt.title(f"Actual: {label.item()} | Predicted: {predicted} ({confidence:.2%})")
plt.show()