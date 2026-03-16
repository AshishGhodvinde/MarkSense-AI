import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import cv2
import numpy as np
import os
from PIL import Image
import glob

class HandwrittenMarksNet(nn.Module):
    """Specialized network for handwritten marks in boxes"""
    def __init__(self):
        super(HandwrittenMarksNet, self).__init__()
        
        # First block - feature extraction
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # Second block - deeper features
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Third block - high-level features
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 3 * 3, 512)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(256, 10)
        
        self.relu = nn.ReLU()

    def forward(self, x):
        # Block 1
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        
        # Block 2
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        
        # Block 3
        x = self.relu(self.bn5(self.conv5(x)))
        x = self.relu(self.bn6(self.conv6(x)))
        x = self.pool3(x)
        
        # Flatten and classify
        x = x.view(-1, 128 * 3 * 3)
        x = self.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        
        return x

def create_handwritten_marks_augmentation():
    """Create augmentations specific to handwritten marks"""
    return transforms.Compose([
        transforms.RandomRotation(15),  # Moderate rotation
        transforms.RandomAffine(0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.RandomPerspective(distortion_scale=0.05, p=0.3),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

def create_synthetic_handwritten_marks():
    """Create synthetic handwritten marks from existing crops"""
    print("Creating synthetic handwritten marks dataset...")
    
    # Look for existing crop images
    crop_files = glob.glob('raw_crop_*.png')
    if not crop_files:
        print("No crop files found. Using standard MNIST augmentation.")
        return None
    
    synthetic_dataset = []
    labels = []
    
    for crop_file in crop_files:
        try:
            img = cv2.imread(crop_file, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
                
            # Apply various augmentations to create synthetic samples
            for i in range(10):  # Create 10 variations per crop
                # Random rotation
                angle = np.random.uniform(-10, 10)
                h, w = img.shape
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                
                # Random scaling
                scale = np.random.uniform(0.8, 1.2)
                new_w = int(w * scale)
                new_h = int(h * scale)
                scaled = cv2.resize(rotated, (new_w, new_h), interpolation=cv2.INTER_AREA)
                
                # Center back to original size
                if new_w != w or new_h != h:
                    canvas = np.zeros((h, w), dtype=np.uint8)
                    x_offset = (w - new_w) // 2
                    y_offset = (h - new_h) // 2
                    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = scaled
                    scaled = canvas
                
                # Random noise
                noise = np.random.normal(0, 5, scaled.shape).astype(np.uint8)
                noisy = np.clip(scaled.astype(int) + noise, 0, 255).astype(np.uint8)
                
                synthetic_dataset.append(noisy)
                # For now, use random labels - in real scenario, these would be manually labeled
                labels.append(np.random.randint(0, 10))
                
        except Exception as e:
            print(f"Error processing {crop_file}: {e}")
            continue
    
    return synthetic_dataset, labels

def train_handwritten_marks_model():
    """Train specialized model for handwritten marks"""
    print("Training specialized handwritten marks model...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create augmentations
    transform_train = create_handwritten_marks_augmentation()
    
    # Load MNIST with enhanced augmentation
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
    
    # Create validation split
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    # Initialize model
    model = HandwrittenMarksNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    # Training parameters
    epochs = 30
    best_val_loss = float('inf')
    patience_counter = 0
    
    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
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
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        train_acc = 100 * correct / total
        val_acc = 100 * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {running_loss/len(train_loader):.4f}, "
              f"Train Acc: {train_acc:.2f}%, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'handwritten_marks_model.pth')
            print(f"New best model saved! Val Loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= 8:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    print(f"Training completed. Best validation loss: {best_val_loss:.4f}")
    return model

if __name__ == "__main__":
    train_handwritten_marks_model()
