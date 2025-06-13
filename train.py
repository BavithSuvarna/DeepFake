# train.py
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import DataLoader, Dataset, random_split, ConcatDataset
import os
from PIL import Image


print("🚀 Training started...")

# Custom Dataset
class FrameDataset(Dataset):
    def __init__(self, root_dir, label, transform=None):
        self.images = []
        for subdir, _, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.images.append(os.path.join(subdir, file))
        self.images.sort()
        self.label = label
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.label

# Transforms
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Load datasets
real_dataset = FrameDataset('frames/real_video', 0, train_transform)
fake_dataset = FrameDataset('frames/fake_video', 1, train_transform)

# Combine datasets
combined_dataset = ConcatDataset([real_dataset, fake_dataset])

# Split into train and validation sets
train_size = int(0.8 * len(combined_dataset))
val_size = len(combined_dataset) - train_size
train_dataset, val_dataset = random_split(combined_dataset, [train_size, val_size])

# Wrap validation dataset with val_transform
val_dataset.dataset.transform = val_transform

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model
from torchvision.models import resnet18, ResNet18_Weights
weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)
model.fc = nn.Linear(model.fc.in_features, 2)
model.to(device)

# Weighted loss for imbalance (adjust based on your actual counts)
class_weights = torch.tensor([1.0, 5639 / 590], dtype=torch.float32).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Training parameters
num_epochs = 20
best_val_accuracy = 0.0
patience = 5
epochs_without_improvement = 0

# Resume logic
start_epoch = 0
checkpoint_path = "model/latest_checkpoint.pth"

if os.path.exists(checkpoint_path):
    print("🔁 Resuming from latest checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    best_val_accuracy = checkpoint['best_val_accuracy']
    start_epoch = checkpoint['epoch'] + 1
    print(f"Resumed from epoch {start_epoch} with val accuracy {best_val_accuracy:.4f}")

# Training loop with validation and early stopping
for epoch in range(start_epoch, num_epochs):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_accuracy = correct / total
    print(f"📊 Epoch {epoch+1}/{num_epochs}, Loss={loss.item():.4f}, Val Accuracy={val_accuracy:.4f}")

    # Save best model
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(model.state_dict(), "model/best_deepfake_model.pth")
        print("✅ Best model saved.")
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1
        print(f"📉 No improvement for {epochs_without_improvement} epoch(s).")

    # Save latest checkpoint after every epoch
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_accuracy': best_val_accuracy
    }, checkpoint_path)

    if epochs_without_improvement >= patience:
        print("⛔ Early stopping triggered.")
        break
