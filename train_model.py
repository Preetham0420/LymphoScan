import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
from transformers import SwinForImageClassification
from torch.cuda.amp import autocast, GradScaler  # Mixed Precision Training
import time

# ✅ CORRECT PATHS
DATA_DIR = r"C:\Users\jatin\LymphoScan_Project\data\train\AUG"
VAL_DIR = r"C:\Users\jatin\LymphoScan_Project\data\validation"
MODEL_PATH = r"C:\Users\jatin\LymphoScan_Project\models\swin_model.pth"

# 🖥️ Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔥 Using device: {device}")

# 🏗️ Optimized Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),  # More variation
    transforms.RandomRotation(10),      # Small rotation variations
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Standard normalization
])

# 📂 Load datasets
train_dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)
val_dataset = datasets.ImageFolder(root=VAL_DIR, transform=transform)

# ✅ VERIFY DATA LOADING
num_classes = len(train_dataset.classes)
print(f"\n📂 Classes Found: {train_dataset.classes} ({num_classes} classes)")

# 🛠️ Optimized DataLoader
BATCH_SIZE = 32  # ✅ Set batch size to 32
NUM_WORKERS = 0  # ✅ Windows Fix: Set to 0 to prevent multiprocessing error

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

# ✅ RESTORED: EXACT MODEL CONFIG USED FOR swin_model_backup.pth!
print("\n🔄 Loading Model...")
model = SwinForImageClassification.from_pretrained(
    "microsoft/swin-tiny-patch4-window7-224",
    num_labels=num_classes,  # ✅ Ensure correct number of classes (10)
    ignore_mismatched_sizes=True  # ✅ Fixes mismatched classifier layer issue
)
model.to(device)
print("✅ Model Loaded Successfully!")

# 🎯 Define Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-4)  # Weight decay for generalization
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.8)  # Reduce LR every 3 epochs

# 🚀 AMP Mixed Precision Training
scaler = GradScaler()

# 🚀 Training Loop
EPOCHS = 10
print("\n🚀 Training Started...")

start_time = time.time()

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    print(f"\n🟢 Epoch {epoch + 1}/{EPOCHS} Started...")

    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        with autocast():  # Mixed precision for speed
            outputs = model(images).logits
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

        # Compute accuracy on training batch
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

        # 🔄 Print batch progress every 10 batches
        if (batch_idx + 1) % 10 == 0:
            print(f"📌 Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}")

    # Update learning rate
    lr_scheduler.step()

    avg_loss = running_loss / len(train_loader)
    train_accuracy = 100. * correct / total
    print(f"\n✅ Epoch {epoch + 1}/{EPOCHS} Finished - Loss: {avg_loss:.4f}, Train Acc: {train_accuracy:.2f}%")

# ⏳ Training time
end_time = time.time()
print(f"\n⏳ Total Training Time: {round((end_time - start_time) / 60, 2)} minutes")

# 💾 Save Model
torch.save(model.state_dict(), MODEL_PATH)
print(f"✅ Model saved at: {MODEL_PATH}")
