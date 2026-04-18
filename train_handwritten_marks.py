import os
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision import datasets, transforms

from train import MnistCNN


class MarksheetDigitDataset(Dataset):
    def __init__(self, root_dir: str, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples: List[Tuple[str, int]] = []

        for label in range(10):
            digit_dir = os.path.join(root_dir, str(label))
            if not os.path.isdir(digit_dir):
                continue

            for filename in os.listdir(digit_dir):
                if filename.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                    # Simple check for corrupted files
                    full_path = os.path.join(digit_dir, filename)
                    if os.path.getsize(full_path) > 10:
                        self.samples.append((full_path, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        img_path, label = self.samples[index]
        try:
            image = Image.open(img_path).convert("L")
            if self.transform is not None:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a dummy zero image if one fails, to keep the batch consistent
            dummy = torch.zeros((1, 28, 28))
            return dummy, label


def create_train_transform():
    return transforms.Compose(
        [
            transforms.Resize((28, 28)),
            transforms.RandomRotation(10),
            transforms.RandomAffine(0, translate=(0.08, 0.08), scale=(0.92, 1.08)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )


def create_eval_transform():
    return transforms.Compose(
        [
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )


def load_training_dataset(extra_data_dir: str = "labeled_digits"):
    train_transform = create_train_transform()
    eval_transform = create_eval_transform()

    mnist_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=train_transform)

    if not os.path.isdir(extra_data_dir):
        print(f"No custom labeled dataset found at '{extra_data_dir}'. Training on MNIST only.")
        return mnist_dataset, None

    custom_dataset = MarksheetDigitDataset(extra_data_dir, transform=train_transform)
    if len(custom_dataset) == 0:
        print(f"'{extra_data_dir}' exists but contains no labeled digit images. Training on MNIST only.")
        return mnist_dataset, None

    print(f"Loaded {len(custom_dataset)} labeled marksheet digit crops from '{extra_data_dir}'.")
    custom_eval = MarksheetDigitDataset(extra_data_dir, transform=eval_transform)
    return ConcatDataset([mnist_dataset, custom_dataset]), custom_eval


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = total_loss / max(len(loader), 1)
    accuracy = 100.0 * correct / max(total, 1)
    return avg_loss, accuracy


def train_handwritten_marks_model():
    print("Training marksheet digit recognizer...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataset, custom_eval_dataset = load_training_dataset()

    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_subset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=128, shuffle=False)

    custom_eval_loader = None
    if custom_eval_dataset is not None and len(custom_eval_dataset) > 0:
        custom_eval_loader = DataLoader(custom_eval_dataset, batch_size=128, shuffle=False)

    model = MnistCNN().to(device)
    if os.path.exists("mnist_model.pth"):
        model.load_state_dict(torch.load("mnist_model.pth", map_location=device))
        print("Loaded existing MNIST weights as initialization.")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0007, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    best_score = float("-inf")
    epochs = 12

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        custom_acc_text = "n/a"
        score = val_acc
        if custom_eval_loader is not None:
            _, custom_acc = evaluate(model, custom_eval_loader, criterion, device)
            custom_acc_text = f"{custom_acc:.2f}%"
            score = custom_acc

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"train_loss={running_loss / max(len(train_loader), 1):.4f} | "
            f"val_acc={val_acc:.2f}% | custom_acc={custom_acc_text}"
        )

        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), "handwritten_marks_model.pth")
            torch.save(model.state_dict(), "mnist_model.pth")
            print("Saved improved weights to handwritten_marks_model.pth and mnist_model.pth")

    print("Training complete.")
    if custom_eval_loader is not None:
        print("Tip: keep adding labeled crops under labeled_digits/0..9 to improve real marksheet accuracy.")


if __name__ == "__main__":
    train_handwritten_marks_model()
