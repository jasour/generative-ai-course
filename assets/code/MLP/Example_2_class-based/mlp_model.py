import torch.nn as nn  # Import PyTorch's neural network module for defining models.

# Define MLP (Multi-Layer Perceptron) Model
class MLP(nn.Module):  # Every PyTorch model must inherit from nn.Module.
    # It initializes the model layers when a new MLP object is created.
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        
        #  Initialize the parent class (nn.Module) inside the child class (MLP). 
        super(MLP, self).__init__()  

        # First fully connected layer: Takes 'input_size' features and outputs 'hidden_size1' features.
        self.fc1 = nn.Linear(input_size, hidden_size1)

        # First activation function: ReLU (Rectified Linear Unit) introduces non-linearity.
        self.relu1 = nn.ReLU()

        # Second fully connected layer: Takes 'hidden_size1' features and outputs 'hidden_size2' features.
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)

        # Second activation function: ReLU introduces non-linearity again.
        self.relu2 = nn.ReLU()

        # Third fully connected layer (output layer): Maps 'hidden_size2' features to 'output_size' classes.
        self.fc3 = nn.Linear(hidden_size2, output_size)

    # Define the forward pass of the network (how data flows through the layers).
    def forward(self, x):
        x = self.fc1(x)   # Pass input through the first fully connected layer.
        x = self.relu1(x) # Apply ReLU activation function.
        x = self.fc2(x)   # Pass output to the second fully connected layer.
        x = self.relu2(x) # Apply ReLU activation function.
        x = self.fc3(x)   # Pass output to the final layer (output layer).
        return x          # Return the final output (logits for classification).