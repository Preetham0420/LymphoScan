import sys
import torch
from PIL import Image
from torchvision import transforms
from transformers import SwinForImageClassification

# ✅ Get image path from command-line argument
if len(sys.argv) > 1:
    IMAGE_PATH = sys.argv[1]
else:
    print("❌ No image path provided! Usage: python predict.py <image_path>")
    sys.exit(1)

# ✅ Load Model
MODEL_PATH = "C:\\Users\\jatin\\LymphoScan_Project\\models\\swin_model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("🔥 Using device:", device)
print("🔄 Loading model...")

try:
    # ✅ Load Swin Transformer model with 10 classes
    model = SwinForImageClassification.from_pretrained(
        "microsoft/swin-tiny-patch4-window7-224",
        num_labels=10,  # ✅ Ensure correct number of classes (10)
        ignore_mismatched_sizes=True
    ).to(device)

    # ✅ Load trained weights
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device), strict=False)
    model.eval()
    print("✅ Model loaded successfully!")

except Exception as e:
    print(f"❌ Error loading model: {e}")
    sys.exit(1)

# ✅ Load and Preprocess Image
try:
    image = Image.open(IMAGE_PATH).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ✅ Match training normalization
    ])
    image = transform(image).unsqueeze(0).to(device)

except Exception as e:
    print(f"❌ Error loading image: {e}")
    sys.exit(1)

# ✅ Predict
with torch.no_grad():
    outputs = model(image)
    _, predicted = torch.max(outputs.logits, 1)

# ✅ Updated Class Labels (Now includes 'non_cancer' & 'unknown')
class_names = ['Benign', 'Blood_Cancer', 'CLL', 'Early', 'FL', 'MCL', 'Pre', 'Pro', 'non_cancer', 'unknown']
predicted_class = class_names[predicted.item()]

print(f"📢 Prediction: {predicted_class}")
