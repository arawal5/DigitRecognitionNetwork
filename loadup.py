import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = torch.device("cuda")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_data = datasets.MNIST(root=".", train=True, download=False, transform=transform)
test_data = datasets.MNIST(root=".", train=False, download=False, transform=transform)
    
# 4. Wrap datasets in DataLoaders for batching
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=1000, shuffle=False)

# 5. Define the Feedforward Neural Network
class FeedforwardMNIST(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.network = nn.Sequential(
            nn.Linear(28 * 28, 256),  # Input: 784 pixels -> 256 neurons
            nn.ReLU(),
            nn.Dropout(0.2),          # Prevents overfitting
            nn.Linear(256, 128),      # 256 -> 128 neurons
            nn.ReLU(),
            nn.Linear(128, 10)        # 128 -> 10 digit classes (0-9)
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.network(x)

model = FeedforwardMNIST().to(device)

# 6. Set up loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 7. Training loop
epochs = 5
for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()            # Clear old gradients
        outputs = model(images)          # Forward pass
        loss = criterion(outputs, labels) # Calculate error
        loss.backward()                  # Backpropagation
        optimizer.step()                 # Update weights

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch + 1}/{epochs}] - Loss: {avg_loss:.4f}")

# 8. Evaluation loop
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")