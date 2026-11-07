import os
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, datasets,transforms
from torch.utils.data import DataLoader

# بندرب هنا الموديل عشان متخاذل و مش بيروح الجيم
# كاريم هيزعل منه كدا فجالي عشان اروق عليه

matplotlib.use('Agg')

# هنا بجيب شوية الصور الحلوة
DATASET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), r"Face Mask Dataset\dataset")
SAVE_PATH    = 'mask_detector.pth'
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Using: {DEVICE}")

# اروق عليها بعد ما عمر ظبطها
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


train_data = datasets.ImageFolder(os.path.join(DATASET_PATH, 'train'),      transform=transform)
val_data   = datasets.ImageFolder(os.path.join(DATASET_PATH, 'val'), transform=transform)
test_data  = datasets.ImageFolder(os.path.join(DATASET_PATH, 'test'),       transform=transform)

CLASSES = train_data.classes
NUM_CLASSES = len(CLASSES)
print(f"Classes found: {CLASSES} (Total: {NUM_CLASSES})")

# هاخد منها شوية كدا صغيرين عشان المساحة متمتش مني
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_data,   batch_size=64)
test_loader  = DataLoader(test_data,  batch_size=64)

print(f"Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")


model = models.mobilenet_v2(weights= 'IMAGENET1K_V1')

for param in model.features[-3:].parameters():
    param.requires_grad = True

# overfitting فقدان ذاكرة كدا لزوم ال 
model.classifier = nn.Sequential(
    nn.Dropout(0.2),
    nn.Linear(model.last_channel, NUM_CLASSES),
)

model = model.to(DEVICE)


loss_fn   = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)


def run_epoch(loader, train=False):

    if train:
        model.train()
    else:
        model.eval()
        
    total_loss = 0
    correct    = 0

    with torch.set_grad_enabled(train):
        for images, labels in loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            predictions = model(images)
            loss        = loss_fn(predictions, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(images)
            correct    += (predictions.argmax(1) == labels).sum().item()

    accuracy = correct / len(loader.dataset)
    avg_loss = total_loss / len(loader.dataset)

    return avg_loss, accuracy


# هنا بقا الشغل كله
# بمرنه باكتير عشر عدات
# و ده طبعا لو موصلش للفشل العضلي ( التسعين فالمية ) 
# و ققل حاجة تلت عدات عشان نلاقي حاجة نرسمها نحت فالجرافات

train_losses = []
val_losses   = []
train_accs   = []
val_accs     = []
best_accs   = 0.0

print("\nTraining...")

# رضا اوي كدا مش ثانوية يعني هيا
ACCURACY_THRESHOLD = 0.90
for epoch in range(1, 11):

    train_loss, train_acc = run_epoch(train_loader, train=True)
    val_loss,   val_acc   = run_epoch(val_loader)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)

    print(f"Epoch {epoch}/10  |  Train Acc: {train_acc:.3f}  |  Val Acc: {val_acc:.3f}")

    # شوفلي كدا جاب كام
    if val_acc >= ACCURACY_THRESHOLD and epoch >= 3:
        print(f"  Target accuracy of {ACCURACY_THRESHOLD*100}% reached! Saving and stopping.")
        torch.save(model.state_dict(), SAVE_PATH)
        break
    
    # لو متفوق هاته
    if val_acc > best_accs:
        best_accs = val_acc
        torch.save(model.state_dict(), SAVE_PATH)
        print(f"  New best model saved (val acc = {best_accs:.3f})")

model.load_state_dict(torch.load(SAVE_PATH))

# شوية رسم لزوم الفن انتا عارف سيف بقا 
# شيف و فنان :)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(train_losses, label='Train', marker='o')
ax1.plot(val_losses,   label='Validation', marker='o')
ax1.set_title('Loss')
ax1.legend()
ax1.grid(alpha=0.3)

ax2.plot(train_accs, label='Train', marker='o')
ax2.plot(val_accs,   label='Validation', marker='o')
ax2.axhline(0.90, color='green', linestyle='--', label='90% target')
ax2.set_title('Accuracy')
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('training_curves.png', dpi=150)
plt.close()

print("Training curves saved to training_curves.png")


_, test_acc = run_epoch(test_loader)

print(f"Test Accuracy : {test_acc * 100:.2f}%")
print(f"Result        : {'PASSED' if test_acc >= 0.90 else 'Below target'}")

# سيفو بقا عندك عشان اللاب هيضربنا لو مرنا عليه كمان مره
torch.save({
    'model_state_dict' : model.state_dict(),
    'class_to_idx'     : train_data.class_to_idx,
    'idx_to_class'     : {v: k for k, v in train_data.class_to_idx.items()},
    'img_size'         : 224,
    'test_accuracy'    : round(test_acc, 4),
}, SAVE_PATH)
