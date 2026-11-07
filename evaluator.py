import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.metrics import (confusion_matrix, precision_score, recall_score, f1_score)
from PIL import Image

# ============================================================
# 1. SETTINGS
# ============================================================
base_path = os.path.dirname(os.path.abspath(__file__))

DATASET_PATH = os.path.join(base_path, r"Face Mask Dataset\dataset")
MODEL_PATH   = os.path.join(base_path, "mask_detector.pth")
CLASSES      = ['WithMask', 'WithoutMask']
IMG_SIZE     = 224
BATCH_SIZE   = 32
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Using device: {DEVICE}")

# ============================================================
# 2. LOAD TEST DATA
# ============================================================
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

test_dataset = ImageFolder(os.path.join(DATASET_PATH, 'test'), transform=transform)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"Test Images: {len(test_dataset)}")
print(f"Classes: {test_dataset.classes}")


model = models.mobilenet_v2(weights=None)
model.classifier = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(model.last_channel, 2)
)
# 1. Load the entire saved dictionary
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

# 2. Extract just the weights and load them into the model
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(DEVICE)
model.eval()
print("Model loaded successfully!")

# ============================================================
# 4. GET PREDICTIONS
# ============================================================
all_preds  = []
all_labels = []
all_paths  = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)
        outputs = model(images)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)

# Also collect image paths for misclassification
for img_path, _ in test_dataset.imgs:
    all_paths.append(img_path)

# ============================================================
# 5. CONFUSION MATRIX
# ============================================================
cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASSES, yticklabels=CLASSES)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig(os.path.join(base_path, 'confusion_matrix.png'))
plt.show()


# ============================================================
# 6. PER-CLASS METRICS
# ============================================================


precision = precision_score(all_labels, all_preds, average=None)
recall    = recall_score(all_labels, all_preds, average=None)
f1        = f1_score(all_labels, all_preds, average=None)
accuracy  = (all_preds == all_labels).mean()

print(f"Overall Accuracy: {accuracy * 100:.2f}%\n")

# Plot per-class metrics bar chart
x = np.arange(len(CLASSES))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(x - width, precision, width, label='Precision', color='steelblue')
ax.bar(x,         recall,    width, label='Recall',    color='orange')
ax.bar(x + width, f1,        width, label='F1-Score',  color='green')

ax.set_xticks(x)
ax.set_xticklabels(CLASSES)
ax.set_ylim(0, 1.1)
ax.set_title('Per-Class Metrics')
ax.set_ylabel('Score')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(base_path, 'per_class_metrics.png'))
plt.show()


# ============================================================
# 7. MISCLASSIFICATION ANALYSIS
# ============================================================
wrong_idx = np.where(all_preds != all_labels)[0]
print(f"\nTotal Misclassified: {len(wrong_idx)} / {len(all_labels)}")

# Show up to 9 misclassified images
n_show = min(9, len(wrong_idx))
fig, axes = plt.subplots(3, 3, figsize=(12, 12))
axes = axes.flatten()

for i in range(n_show):
    idx        = wrong_idx[i]
    img        = Image.open(all_paths[idx]).convert('RGB')
    true_label = CLASSES[all_labels[idx]]
    pred_label = CLASSES[all_preds[idx]]

    axes[i].imshow(img.resize((IMG_SIZE, IMG_SIZE)))
    axes[i].set_title(f'True: {true_label}\nPred: {pred_label}', color='red')
    axes[i].axis('off')

# Hide unused axes
for j in range(n_show, len(axes)):
    axes[j].axis('off')

plt.suptitle('Misclassified Images', fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(base_path, 'misclassifications.png'))
plt.show()


