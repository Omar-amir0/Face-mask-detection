import os
import random
import shutil
from PIL import Image

random.seed(42)

path = os.path.dirname(os.path.abspath(__file__))

SOURCE_DIR = os.path.join(path, r"Face Mask Dataset\Train")
DEST_DIR = os.path.join(path, r"Face Mask Dataset\dataset")

splits = (0.7, 0.15, 0.15)

#Clean BEFORE splitting
def clean_dataset(folder):
    for root, _, files in os.walk(folder):
        for file in files:
            path = os.path.join(root, file)
            try:
                img = Image.open(path)
                img.verify()
            except:
                print("Removing corrupted:", path)
                os.remove(path)

clean_dataset(SOURCE_DIR)

# Split dataset
for cls in os.listdir(SOURCE_DIR):
    cls_path = os.path.join(SOURCE_DIR, cls)

    if not os.path.isdir(cls_path):
        continue

    images = [img for img in os.listdir(cls_path) 
              if img.lower().endswith(('.png', '.jpg', '.jpeg'))]

    random.shuffle(images)

    train_end = int(len(images) * splits[0])
    val_end = int(len(images) * (splits[0] + splits[1]))

    splits_data = {
        "train": images[:train_end],
        "val": images[train_end:val_end],
        "test": images[val_end:]
    }

    for split, imgs in splits_data.items():
        split_dir = os.path.join(DEST_DIR, split, cls)
        os.makedirs(split_dir, exist_ok=True)

        for img in imgs:
            src = os.path.join(cls_path, img)
            dst = os.path.join(split_dir, img)
            shutil.copy2(src, dst)

print("Dataset organized successfully")

# Check empty folders
for root, dirs, files in os.walk(DEST_DIR):
    if len(files) == 0:
        print("Empty folder:", root)

#Count images correctly
for split in ["train", "val", "test"]:
    print(f"\n{split.upper()}")
    path = os.path.join(DEST_DIR, split)
    
    for cls in os.listdir(path):
        count = len(os.listdir(os.path.join(path, cls)))
        print(f"{cls}: {count}")