import os
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms as T, models
from PIL import Image
from pathlib import Path

# === 1. Setup base path ===
BASE_DIR = Path(__file__).resolve().parent
image_dir = BASE_DIR / "data" / "extracted_frames"  # Use relative path

# === 2. Dataset Definition ===
class DeepfakeDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        label = 1 if 'FAKE' in img_name.upper() else 0
        img_path = self.root_dir / img_name
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)

# === 3. Transformations ===
transform = T.Compose([
    T.Resize((128, 128)),
    T.RandomHorizontalFlip(),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    T.RandomRotation(10),
    T.ToTensor(),
])

# === 4. Load Dataset ===
dataset = DeepfakeDataset(image_dir, transform=transform)

# === 5. Train/Val Split ===
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# === 6. Model Setup (Transfer Learning) ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.resnet18(pretrained=True)

# Freeze base layers
for param in model.parameters():
    param.requires_grad = False

# Unfreeze only final layers
for name, param in model.named_parameters():
    if 'layer4' in name or 'fc' in name:
        param.requires_grad = True

model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)

# === 7. Optimizer & Loss ===
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.5)

# === 8. Training Loop ===
print("🚀 Training started...")
best_val_acc = 0

for epoch in range(20):
    model.train()
    total_loss = 0
    correct = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()

    train_acc = correct / len(train_dataset) * 100

    # === Validation ===
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for val_images, val_labels in val_loader:
            val_images, val_labels = val_images.to(device), val_labels.to(device)
            val_outputs = model(val_images)
            val_correct += (val_outputs.argmax(1) == val_labels).sum().item()
            val_total += val_labels.size(0)

    val_acc = val_correct / val_total * 100
    scheduler.step(val_acc)

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        joblib.dump(model, BASE_DIR / "deepfake_model.pkl")
        print("✅ Best model saved as deepfake_model.pkl")

print(f"🏁 Training complete. Best Validation Accuracy: {best_val_acc:.2f}%")
