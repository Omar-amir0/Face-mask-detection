
# =========================
# 1. Imports #eda notebook
# =========================
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from collections import Counter

# =========================
# 2. Dataset Path
# =========================
path = os.path.dirname(os.path.abspath(__file__))

# This attaches 'Dataset' to that folder path
DATASET_PATH = os.path.join(path, "Face Mask Dataset/Test")
CLASSES = ['WithMask', 'WithoutMask']

# =========================
# 3. Load Dataset Metadata
# =========================
data = []
for label in CLASSES:
    folder = os.path.join(DATASET_PATH, label)
    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        try:
            img = Image.open(path)
            width, height = img.size
            arr = np.array(img)
            brightness = arr.mean()
            contrast = arr.std()
            data.append({
                'path': path,
                'label': label,
                'width': width,
                'height': height,
                'aspect_ratio': width / height,
                'brightness': brightness,
                'contrast': contrast
            })
        except:
            continue

_df = pd.DataFrame(data)
print(_df.head())
print(f'Total Images: {len(_df)}')

# =========================
# 4. Sample Grid Function
# =========================
def show_samples(label, n=9):
    samples = _df[_df['label']==label]['path'].sample(n)
    plt.figure(figsize=(10,10))
    for i, path in enumerate(samples):
        img = Image.open(path)
        plt.subplot(3,3,i+1)
        plt.imshow(img)
        plt.axis('off')
    plt.suptitle(label)
    plt.show()

# Run for each class
for c in CLASSES:
    show_samples(c)

# =========================
# 5. Class Distribution
# =========================
sns.countplot(data=_df, x='label')
plt.title('Class Distribution')
plt.show()

# =========================
# 6. Width / Height Distribution
# =========================
_df['width'].hist(bins=30)
plt.title('Width Distribution')
plt.show()

_df['height'].hist(bins=30)
plt.title('Height Distribution')
plt.show()

# =========================
# 7. Aspect Ratio Distribution
# =========================
_df['aspect_ratio'].hist(bins=30)
plt.title('Aspect Ratio Distribution')
plt.show()

# =========================
# 8. Brightness Distribution
# =========================
sns.histplot(_df['brightness'], bins=30)
plt.title('Brightness Distribution')
plt.show()

# =========================
# 9. Contrast Distribution
# =========================
sns.histplot(_df['contrast'], bins=30)
plt.title('Contrast Distribution')
plt.show()

# =========================
# 10. Correlation Heatmap
# =========================
sns.heatmap(_df[['width','height','aspect_ratio','brightness','contrast']].corr(), annot=True)
plt.title('Feature Correlation')
plt.show()

# =========================
# 11. Summary Statistics
# =========================
print(_df.groupby('label')[['width','height','brightness','contrast']].mean())

