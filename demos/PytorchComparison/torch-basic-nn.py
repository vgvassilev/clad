# Do the same as torch-basic-nn.cpp but in Python using PyTorch
# To run: python torch-basic-nn.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

M = 100 # Number of samples
N = 100   # Number of features
N_ITER = 5000 # Number of iterations

# Create a simple neural network with 3 hidden nodes and 1 output node
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N, 3)
        self.fc2 = nn.Linear(3, 1)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def main():
    # Create a simple dataset
    x = np.random.randn(M, N).astype(np.float32)
    y = np.random.randn(M, 1).astype(np.float32)

    # Convert the dataset to PyTorch tensors
    x = torch.from_numpy(x)
    y = torch.from_numpy(y)

    # Create the neural network
    net = Net()

    # Create the loss function and the optimizer
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01)

    # Print the initial loss
    output = net(x)
    loss = criterion(output, y)
    print("Initial loss: ", loss.item())

    # Calculate start time
    import time
    start = time.time()

    # Train the neural network
    for i in range(N_ITER):
        optimizer.zero_grad()
        output = net(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

    # Stop the timer
    end = time.time()

    # Print the final loss
    output = net(x)
    loss = criterion(output, y)
    print("Final loss: ", loss.item())
    print("Time: ", end - start, " seconds")
    

if __name__ == "__main__":
    main()
