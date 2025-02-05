 import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader

# Load dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
trainloader = DataLoader(FashionMNIST("./data", train=True, download=True, transform=transform), batch_size=64, shuffle=True)

testloader = DataLoader(FashionMNIST("./data", train=False, download=True, transform=transform), batch_size=64, shuffle=False)

# Define model
class SoftmaxRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(28*28, 10)
    def forward(self, x):
        return self.linear(x.view(-1, 28*28))

# Initialize model, loss, optimizer
model = SoftmaxRegression()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train model
def train_model(model, trainloader, criterion, optimizer, epochs=5):
    for epoch in range(epochs):
        for images, labels in trainloader:
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

train_model(model, trainloader, criterion, optimizer)

