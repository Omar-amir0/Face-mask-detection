import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from PIL import Image

torch.manual_seed(42)
np.random.seed(42)

# Configuration
path = os.path.dirname(os.path.abspath(__file__))

# This attaches 'Dataset' to that folder path
DATASET_PATH = os.path.join(path, "Face Mask Dataset/dataset")
CLASSES = ['WithMask', 'WithoutMask']
IMG_SIZE = 224
BATCH_SIZE = 32

# Base transform (no augmentation)
transform_no_aug = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]),
])

# Augmentation pipeline
transform_aug = T.Compose([
    T.Resize((IMG_SIZE + 20, IMG_SIZE + 20)),
    T.RandomCrop((IMG_SIZE, IMG_SIZE)),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(10),
    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
    T.RandomGrayscale(p=0.05),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]),
])

transform_val = transform_no_aug

# Load datasets
train_aug    = ImageFolder(os.path.join(DATASET_PATH, 'train'), transform=transform_aug)
train_noaug  = ImageFolder(os.path.join(DATASET_PATH, 'train'), transform=transform_no_aug)
val_dataset  = ImageFolder(os.path.join(DATASET_PATH, 'val'),   transform=transform_val)
test_dataset = ImageFolder(os.path.join(DATASET_PATH, 'test'),  transform=transform_val)

# Data loaders
train_aug_loader   = DataLoader(train_aug,    batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
train_noaug_loader = DataLoader(train_noaug,  batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
val_loader         = DataLoader(val_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
test_loader        = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

print("Train (Aug):", len(train_aug))
print("Validation:", len(val_dataset))
print("Test:", len(test_dataset))
print("Classes:", train_aug.classes)


# Visualization
def show_augmentation_examples(dataset_path, label='WithMask', n=8):
    folder = os.path.join(dataset_path, 'train', label)
    img_path = os.path.join(folder, os.listdir(folder)[0])
    img = Image.open(img_path).convert('RGB')

    aug = T.Compose([
        T.Resize((IMG_SIZE + 20, IMG_SIZE + 20)),
        T.RandomCrop((IMG_SIZE, IMG_SIZE)),
        T.RandomHorizontalFlip(),
        T.RandomRotation(10),
        T.ColorJitter(0.3, 0.3, 0.2, 0.05),
        T.RandomGrayscale(0.05),
    ])

    fig, axes = plt.subplots(2, n // 2 + 1, figsize=(16, 6))

    axes[0][0].imshow(img.resize((IMG_SIZE, IMG_SIZE)))
    axes[0][0].set_title('Original')
    axes[0][0].axis('off')
    axes[1][0].axis('off')

    for i in range(n):
        r, c = divmod(i, n // 2)
        ax = axes[r][c + 1]
        ax.imshow(aug(img))
        ax.set_title(f'Aug {i+1}')
        ax.axis('off')

    plt.tight_layout()
    plt.show()


# Run example
show_augmentation_examples(DATASET_PATH, label='WithMask')