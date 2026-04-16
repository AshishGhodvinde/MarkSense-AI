import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import os
import random
import numpy as np

# We import the exact model being used by the processor
from train import MnistCNN

class RandomGridLines(object):
    """Simulates black grid lines over the white digit (after ToTensor, MNIST is black background but our crops are inverted so let's stick to the tensor format)
    MNIST tensors are 1 channel, normalized. E.g. background is ~ -0.4, digit is ~ 2.8
    """
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, tensor):
        if random.random() < self.p:
            h, w = tensor.shape[1], tensor.shape[2]
            # Add 1 to 2 random horizontal lines
            for _ in range(random.randint(0, 2)):
                y = random.randint(0, h-1)
                thickness = random.randint(1, 2)
                tensor[:, y:min(h, y+thickness), :] = 2.0 # high intensity to simulate thick lines
                
            # Add 1 to 2 random vertical lines
            for _ in range(random.randint(0, 2)):
                x = random.randint(0, w-1)
                thickness = random.randint(1, 2)
                tensor[:, :, x:min(w, x+thickness)] = 2.0
        return tensor

def train_augmented_model():
    print("Preparing massively augmented dataset...")
    
    # Aggressive augmentations to mimic noisy camera capture of cells
    transform_train = transforms.Compose([
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5), # Severe skew
        transforms.RandomRotation(20),  # heavier rotation
        transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), scale=(0.8, 1.2), shear=10),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        RandomGridLines(p=0.6) # 60% chance of grid lines
    ])
    
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = MnistCNN().to(device)
    
    # Load existing weights if they exist to continue training rather than starting from scratch
    if os.path.exists('mnist_model.pth'):
        print("Loading existing mnist_model.pth as base...")
        model.load_state_dict(torch.load('mnist_model.pth', map_location=device))
        
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4) # AdamW works better
    # Step down LR every 10 epochs
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    epochs = 40  # 40 more epochs
    print(f"Deep Training started for {epochs} epochs...")
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if (i+1) % 150 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}, Acc: {100 * correct / total:.2f}%")
                
        scheduler.step()
        
        # Save every epoch
        model_path = 'mnist_model.pth'
        torch.save(model.state_dict(), model_path)
    
    print(f"Model Training completed aggressively and saved to {model_path}!")

if __name__ == "__main__":
    train_augmented_model()
