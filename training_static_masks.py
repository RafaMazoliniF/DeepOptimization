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

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# >>> FIX 1: Create a generator for the specific device
generator = torch.Generator(device='cuda')

# >>> FIX 2: Pass the generator to the DataLoader
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, generator=generator)
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
def test_model(test_loader, model, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    model.train()
    return 100 * correct / total

# Initialize model and training components
selection_mask = sm.SelectionMask(shape=(1,28,28)).to(device) # Move mask to device
model = SimpleCNN(selection_mask).to(device)
criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

# Create checkpoints directory
os.makedirs('trainings/1', exist_ok=True)

# Training loop
epochs = 300
s = 0.1
last_loss = 0
adapt_lambda = False

for epoch in range(epochs):
    print(f"\nEpoch: {epoch}")
    running_loss = 0.0
    
    if adapt_lambda:
        s = s * 1.1
    
    for images, labels in train_loader:
        # Move data to device
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels) + (s * sm.mask_l1_loss(model.mask)) # combined loss
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    avg_loss = running_loss / len(train_loader)
    
    if last_loss * 0.9 <= avg_loss <= last_loss * 1.1:
        adapt_lambda = True
    else: adapt_lambda = True
    last_loss = avg_loss
        
    # Save checkpoint 
    accuracy = test_model(test_loader, model, device)
    print(f"Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    
    checkpoint = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
        'accuracy': accuracy,
        's': s
    }
    torch.save(checkpoint, f'trainings/1/checkpoint_epoch_{epoch + 1}.pt')

# Save final model
final_accuracy = test_model(test_loader, model, device)
print(f"\nFinal Accuracy: {final_accuracy:.2f}%")
torch.save({
    'epoch': epochs,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': avg_loss,
    'accuracy': final_accuracy,
    's': s
}, 'trainings/1/final_model.pt')