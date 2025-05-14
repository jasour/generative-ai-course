import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
from main_mlp_train import mlp, model_arch, input_size, device  # 

# Load model
model = mlp(model_arch, activation=torch.nn.ReLU, output_activation=torch.nn.Identity)
model.load_state_dict(torch.load("mlp_model.pth", map_location=device))
model.to(device)
model.eval()

# Load one test image
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)
image, label = next(iter(test_loader))
image_flat = image.view(-1, input_size).to(device)

# Predict
with torch.no_grad():
    output = model(image_flat)
    probs = F.softmax(output, dim=1)
    predicted = torch.argmax(probs, dim=1).item()
    confidence = probs.max().item()

# Display result
print(f"Actual: {label.item()} | Predicted: {predicted} (Confidence: {confidence:.2%})")
plt.imshow(image.squeeze(), cmap="gray")
plt.title(f"Actual: {label.item()} | Predicted: {predicted}")
plt.axis("off")
plt.show()