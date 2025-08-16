import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from transformers import SwinForImageClassification
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# ✅ Paths
VAL_DIR = r"C:\Users\jatin\LymphoScan_Project\data\validation"
MODEL_PATH = r"C:\Users\jatin\LymphoScan_Project\models\swin_model.pth"

# 🖥️ Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔥 Using device: {device}")

# 🏗️ Define same transforms as training for consistency
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Same as training
])

# 📂 Load validation dataset
val_dataset = datasets.ImageFolder(root=VAL_DIR, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

# ✅ Load class names (order is important)
class_names = val_dataset.classes
num_classes = len(class_names)
print(f"\n📂 Classes Found in Validation: {class_names} ({num_classes} classes)")

# 🔄 Load Model
print("\n🔄 Loading Model...")
model = SwinForImageClassification.from_pretrained(
    "microsoft/swin-tiny-patch4-window7-224",
    num_labels=num_classes,  # ✅ Ensure correct number of classes
    ignore_mismatched_sizes=True
)
model.to(device)

# ✅ Load trained weights
try:
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    print("✅ Model Loaded Successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit()

# 🔍 Evaluate Model
model.eval()
true_labels = []
pred_labels = []

with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images).logits
        _, predicted = torch.max(outputs, 1)

        true_labels.extend(labels.cpu().numpy())
        pred_labels.extend(predicted.cpu().numpy())

# 📊 Generate Evaluation Metrics
print("\n📊 Classification Report:")
print(classification_report(true_labels, pred_labels, target_names=class_names, digits=4))

# 🔄 Confusion Matrix
conf_matrix = confusion_matrix(true_labels, pred_labels)
print("\n📌 Confusion Matrix:")
print(conf_matrix)
