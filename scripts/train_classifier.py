# scripts/train_classifier.py
"""
Train a clothing-type classifier on DeepFashion2 subset.
Compatible with Windows multiprocessing and PyTorch >= 0.13.
"""

import os
import torch
from torchvision import transforms, datasets, models
from torch import nn, optim
from torch.utils.data import DataLoader
from datetime import datetime
from tqdm import tqdm

# -------------------------
# Paths
# -------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data_subset")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# -------------------------
# Training Function
# -------------------------
def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🟢 Using device: {device}")

    # transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # datasets
    train_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=transform)
    val_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "val"), transform=transform)

    # dataloaders (Windows-safe)
    train_dl = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0)

    # model
    weights = models.ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, len(train_ds.classes))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    print(f"🧠 Training on {len(train_ds.classes)} classes: {train_ds.classes}")

    # training loop
    for epoch in range(3):
        model.train()
        running_corrects, total = 0, 0
        epoch_loss = 0

        for imgs, labels in tqdm(train_dl, desc=f"Epoch {epoch+1}"):
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels).item()
            total += labels.size(0)
            epoch_loss += loss.item()

        acc = running_corrects / total
        print(f"Epoch {epoch+1} ✅ Loss: {epoch_loss/len(train_dl):.4f}, Accuracy: {acc:.3f}")

    # save
    save_path = os.path.join(MODEL_DIR, f"cloth_classifier_{datetime.now():%Y%m%d_%H%M}.pth")
    torch.save({
        "model": model.state_dict(),
        "classes": train_ds.classes
    }, save_path)

    print(f"\n✅ Model saved to: {save_path}")
    print("Classes:", train_ds.classes)


# -------------------------
# Windows-safe entry point
# -------------------------
if __name__ == "__main__":
    train_model()
