import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import SelectionMask as sm

# Setup data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0,), (1,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Model definition
class SimpleCNN(nn.Module):
    def __init__(self, mask):
        super(SimpleCNN, self).__init__()
        self.mask = mask # define mask
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding="same")
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding="same")
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding="same")
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128 * 3 * 3, 512)
        self.fc2 = nn.Linear(512, 10)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.mask(x) # apply mask
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.dropout1(x)
        x = x.view(-1, 128 * 3 * 3)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

# Test function
def test_model(test_loader, model):
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

# Initialize model and training components
selection_mask = sm.SelectionMask(shape=(1,28,28))
model = SimpleCNN(selection_mask)
criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Create checkpoints directory
os.makedirs('checkpoints', exist_ok=True)

# Training loop
epochs = 300
s = 0.1

for epoch in range(epochs):
    print("\nepoch:", epoch)
    running_loss = 0.0
    s = s * 1.1
    
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels) + (s * sm.mask_l1_loss(model.mask)) # combined loss
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    avg_loss = running_loss / len(train_loader)
        
    # Save checkpoint 
    accuracy = test_model(test_loader, model)
    checkpoint = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
        'accuracy': accuracy,
        's': s
    }
    torch.save(checkpoint, f'checkpoints/checkpoint_epoch_{epoch + 1}.pt')

# Save final model
final_accuracy = test_model(test_loader, model)
torch.save({
    'epoch': epochs,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': avg_loss,
    'accuracy': final_accuracy,
    's': s
}, 'checkpoints/final_model.pt')