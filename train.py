import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.optim as optim # type: ignore
from torch.utils.data import Dataset, DataLoader # type: ignore
from torchvision import transforms, models # type: ignore
import os
import pandas as pd # type: ignore
import numpy as np # type: ignore
from PIL import Image # type: ignore

class HFMnistDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # The parquet dataset from HF usually has 'image' and 'label' columns
        # 'image' might be a dict with 'bytes' or a flat array depending on how pandas reads it
        row = self.df.iloc[idx]
        img_data = row['image']
        
        # Handle the HF image format (often a dict with 'bytes' or 'path')
        if isinstance(img_data, dict) and 'bytes' in img_data:
            import io
            image = Image.open(io.BytesIO(img_data['bytes'])).convert('L')
        else:
            # Fallback for different parquet structures (e.g. flat pixel array)
            if isinstance(img_data, (list, np.ndarray)):
                image = Image.fromarray(np.array(img_data).reshape(28, 28).astype(np.uint8)).convert('L')
            else:
                # If it's already an image-like object or needs custom handling
                try:
                    # Some parquet readers return a flattened version of the image
                    # Let's try to infer if it's a 784-length array
                    flat_data = np.array(img_data)
                    if flat_data.size == 784:
                        image = Image.fromarray(flat_data.reshape(28, 28).astype(np.uint8)).convert('L')
                    else:
                        image = Image.fromarray(np.array(img_data).astype(np.uint8)).convert('L')
                except:
                    # Final fallback: just try to convert to Image
                    image = Image.fromarray(np.array(img_data)).convert('L')

        if self.transform:
            image = self.transform(image)
        
        label = int(row['label'])
        return image, label

class MnistResNet(nn.Module):
    def __init__(self):
        super(MnistResNet, self).__init__()
        # Load a pretrained ResNet18 and modify it for 1-channel grayscale input
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 10)

    def forward(self, x):
        return self.resnet(x)

# Keeping MnistCNN for backward compatibility if needed, but we will use MnistResNet
class MnistCNN(nn.Module):
    def __init__(self):
        super(MnistCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, 128 * 3 * 3)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def train_model():
    print("Downloading/Loading HF MNIST dataset...")
    try:
        # Using a more direct URL format for parquet if the hf:// protocol fails in some environments
        url = "https://huggingface.co/datasets/ylecun/mnist/resolve/main/mnist/train-00000-of-00001.parquet"
        df = pd.read_parquet(url)
        print(f"Dataset loaded: {len(df)} samples.")
    except Exception as e:
        print(f"Failed to load HF dataset: {e}. Falling back to standard torchvision dataset.")
        from torchvision import datasets
        train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=None)
        # We would need to convert this to a DataFrame to maintain the HFMnistDataset logic
        # but for simplicity in fallback, let's just use the standard torchvision loader.
        transform_train = transforms.Compose([
            transforms.RandomRotation(25),
            transforms.RandomAffine(0, translate=(0.2, 0.2), scale=(0.7, 1.3)),
            transforms.RandomPerspective(distortion_scale=0.3, p=0.6),
            transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
        train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        _run_training_loop(train_loader, device)
        return

    print("Preparing dataset with advanced augmentation...")
    transform_train = transforms.Compose([
        transforms.RandomRotation(25),
        transforms.RandomAffine(0, translate=(0.2, 0.2), scale=(0.7, 1.3)),
        transforms.RandomPerspective(distortion_scale=0.3, p=0.6),
        transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = HFMnistDataset(df, transform=transform_train)
    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    _run_training_loop(train_loader, device)

def _run_training_loop(train_loader, device):
    model = MnistCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.003, steps_per_epoch=len(train_loader), epochs=20)
    
    epochs = 5
    print(f"Quick training started ({epochs} epochs)...")
    for epoch in range(epochs):
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            if (i+1) % 200 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

    model_path = 'mnist_model.pth'
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_model()
