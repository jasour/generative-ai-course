import pickle
import torchvision.datasets as datasets

# Define the correct save path
save_path = "deepul/homeworks/hw2/data/cifar10.pkl"

# Download CIFAR-10
trainset = datasets.CIFAR10(root="./data", train=True, download=True)
testset = datasets.CIFAR10(root="./data", train=False, download=True)

# Extract the image arrays
train_data = trainset.data  # Shape: (50000, 32, 32, 3)
test_data = testset.data    # Shape: (10000, 32, 32, 3)

# Save as a dictionary instead of a tuple
with open(save_path, "wb") as f:
    pickle.dump({"train": train_data, "test": test_data}, f)

print(f"✅ Successfully saved CIFAR-10 dataset to {save_path} as a dictionary.")