import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import SelectionMask as sm  # Custom module
import numpy as np
import argparse # To configure via command line


# =============================================================================
# 1. MODEL AND HELPER CLASS DEFINITIONS
# (Well-defined classes can be grouped together)
# =============================================================================

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet_CIFAR(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet_CIFAR, self).__init__()
        self.in_planes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def resnet20(num_classes=10):
    return ResNet_CIFAR(BasicBlock, [3, 3, 3], num_classes=num_classes)

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
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

class LambdaScheduler:
    def __init__(self, init, factor, patience, treshold):
        self.lbd = init
        self.factor = factor
        self.patience = patience
        self.treshold = treshold
        self.limit = np.inf
        self.count = 0
        
    def adapt_lambda(self, new_loss): 
        if new_loss < self.limit:
            self.count = 0
            self.limit = new_loss * (1 - self.treshold)
        else:
            if self.count >= self.patience:
                self.lbd *= self.factor
                self.limit = new_loss * (1 - self.treshold)
                self.count = 0
            else:
                self.count += 1

# =============================================================================
# 2. DATA AND TRAINING FUNCTIONS
# (Separates logic from the main flow)
# =============================================================================

def get_data_loaders(batch_size, validation_split=0.2, seed=42):
    """Encapsulates all data preparation logic."""
    print("Loading MNIST data...")
    cifar_mean = (0.4914, 0.4822, 0.4465)
    cifar_std  = (0.2470, 0.2435, 0.2616)

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cifar_mean, cifar_std),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar_mean, cifar_std),
    ])
        
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    generator = torch.Generator(device=device).manual_seed(seed)

    train_dataset = datasets.CIFAR10(root='./data/cifar', train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR10(root='./data/cifar', train=False, download=True, transform=test_transform)

    n_val = int(len(train_dataset) * validation_split)
    n_train = len(train_dataset) - n_val
    train_subset, val_subset = random_split(train_dataset, [n_train, n_val], generator=generator)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

def test_model(loader, model, device):
    """Generic test function."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            _, predicted = torch.max(y_pred, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    model.train()
    return 100 * correct / total

def train_one_epoch(model, mask_model, loader, criterion, optimizer, scheduler, device):
    """Executes a single training epoch."""
    running_model_loss = 0.0
    running_mask_loss = 0.0
    running_total_loss = 0.0
    
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        
        optimizer.zero_grad()
        
        X_masked = mask_model(X)
        y_pred = model(X_masked)
        
        model_loss = criterion(y_pred, y)
        mask_loss = sm.mask_l1_loss(mask_model)
        loss = model_loss + scheduler.lbd * mask_loss
        
        loss.backward()
        optimizer.step()
        
        running_model_loss += model_loss.item()
        running_mask_loss += mask_loss.item()
        running_total_loss += loss.item()
    
    avg_model_loss = running_model_loss / len(loader)
    avg_mask_loss = running_mask_loss / len(loader)
    avg_total_loss = running_total_loss / len(loader)
    
    return avg_total_loss, avg_model_loss, avg_mask_loss

def training_loop(config, optimizer_class=optim.Adam):
    """Orchestrates the entire training process."""
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data
    train_loader, val_loader, _ = get_data_loaders(config['batch_size'])
    
    # Models and training components
    model = resnet20(num_classes=10).to(device)
    mask_model = sm.SelectionMask(shape=(3, 32, 32)).to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer_kwargs = {}
    if optimizer_class == optim.SGD:
        optimizer_kwargs['momentum'] = 0.9
    
    # Instancia o otimizador
    optimizer = optimizer_class(
        [
            {'params': model.parameters(), 'lr': config['model_learning_rate']},
            {'params': mask_model.parameters(), 'lr': config['mask_learning_rate']}
        ],
        **optimizer_kwargs
    )
    
    scheduler = LambdaScheduler(init=config['lambda_init'], 
                                factor=config['lambda_factor'], 
                                patience=config['lambda_patience'], 
                                treshold=config['lambda_treshold'])
    
    # Create directory to save checkpoints
    training_id = config['training_id']
    checkpoint_dir = f"trainings/{training_id}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Training Loop
    for epoch in range(config['n_epochs']):
        print(f"\n--- Epoch: {epoch+1}/{config['n_epochs']} ---")
        model.train()
        
        avg_total_loss, avg_model_loss, avg_mask_loss = train_one_epoch(
            model, mask_model, train_loader, criterion, optimizer, scheduler, device
        )
        scheduler.adapt_lambda(avg_total_loss)
        
        # Validation
        accuracy = test_model(val_loader, model, device)
        print(f"Total Loss: {avg_total_loss:.4f}, Model Loss: {avg_model_loss:.4f}, "
              f"Mask Loss: {avg_mask_loss:.4f}, Validation Accuracy: {accuracy:.2f}%",
              f"lambda: {scheduler.lbd}, count: {scheduler.count}, menor: {avg_total_loss < scheduler.limit}, limit: {scheduler.limit}")
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'mask_state_dict': mask_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'total_loss': avg_total_loss,
            'model_loss': avg_model_loss,
            'mask_loss': avg_mask_loss,
            'accuracy': accuracy,
            'lambda': scheduler.lbd
        }
        torch.save(checkpoint, f'{checkpoint_dir}/checkpoint_epoch_{epoch + 1}.pt')
        
    print("\nTraining finished!")

# =============================================================================
# 3. MAIN EXECUTION BLOCK
# (Controls the flow and configurations)
# =============================================================================

if __name__ == '__main__':
    # Centralize all configurations here
    CONFIG = {
        "n_epochs": 50,
        "batch_size": 128,
        "model_learning_rate": 0.00001,
        "mask_learning_rate": 0.1,
        "lambda_init": 0.01,
        "lambda_factor": 1.5,
        "lambda_patience": 5,
        "lambda_treshold": 0.2,
        "training_id": "resNet_cifar10_run_ADAM_01"
    }
    
    # Start training
    training_loop(CONFIG, optimizer_class=optim.Adam)