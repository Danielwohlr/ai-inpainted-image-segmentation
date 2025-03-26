import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from loss import HybridLoss, dice_coefficient
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

from unified_dataset import UnifiedDataset

from models.unet_model import get_unet_model
from models.transformer_model import get_transformer_model

# ====================
# Hyperparameters
# ====================
BATCH_SIZE = 128
EPOCHS = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PATIENCE = 10
LR = 1e-3  # Single LR


# ====================
# Albumentations Transform
# ====================
train_transform = A.Compose(
    [
        A.RandomCrop(height=224, width=224, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.CoarseDropout(
            num_holes_range=(3, 8),
            hole_height_range=(10, 32),
            hole_width_range=(10, 32),
            fill=0,
            p=0.5,
        ),
        ToTensorV2(),
    ],
    additional_targets={"mask": "mask"},
)

# ====================
# Datasets & Weighted Sampler
# ====================
# 1) Build unified dataset of manipulated + original
full_dataset = UnifiedDataset(
    manip_image_dir="processed/train/images",
    manip_mask_dir="processed/train/masks",
    originals_dir="processed/train/originals",  # if you have it
    transform=train_transform,
)

# 2) Train/Val split
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# 3) WeightedRandomSampler to get ~10% original (label==1), 90% manipulated (label==0)
train_labels = full_dataset.get_labels()  # all labels from the entire dataset
train_indices = train_dataset.indices  # the subset of indices in train
subset_labels = [
    train_labels[i] for i in train_indices
]  # the 0/1 labels for this train subset

num_orig = sum(x == 1 for x in subset_labels)
num_manip = len(subset_labels) - num_orig

# desired ratio: 0.1 for original, 0.9 for manipulated
weight_orig = 0.1 / (num_orig + 1e-8)
weight_manip = 0.9 / (num_manip + 1e-8)

weights = [weight_orig if lbl == 1 else weight_manip for lbl in subset_labels]

sampler = WeightedRandomSampler(
    weights=weights,
    num_samples=len(train_dataset),  # or bigger if you want
    replacement=True,
)

# 4) DataLoaders
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=sampler,  # ensures ~10% originals on average
    num_workers=8,
    pin_memory=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,  # normal val
    num_workers=8,
    pin_memory=True,
)

# ====================
# Model Creation
# ====================
if False:
    model = get_unet_model().to(DEVICE)
else:
    model = get_transformer_model().to(DEVICE)

criterion = HybridLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

best_val_loss = float("inf")
best_val_dice = 0.0
patience_counter = 0

# ====================
# Training Loop
# ====================
for epoch in range(EPOCHS):
    # Train
    model.train()
    train_loss, train_dice = 0.0, 0.0

    for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"):
        images = images.to(DEVICE, non_blocking=True)
        masks = masks.to(DEVICE, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_dice += dice_coefficient(outputs, masks)

    train_loss /= len(train_loader)
    train_dice /= len(train_loader)

    # Validate
    model.eval()
    val_loss, val_dice_score = 0.0, 0.0
    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):
            images = images.to(DEVICE, non_blocking=True)
            masks = masks.to(DEVICE, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, masks)

            val_loss += loss.item()
            val_dice_score += dice_coefficient(outputs, masks)

    val_loss /= len(val_loader)
    val_dice_score /= len(val_loader)

    print(
        f"Epoch {epoch+1} "
        f"Train Loss: {train_loss:.4f} | Train Dice: {train_dice:.4f} "
        f"| Val Loss: {val_loss:.4f} | Val Dice: {val_dice_score:.4f} "
        f"| Best val loss: {best_val_loss:.4f}"
    )

    # Early stopping (tracking val_loss or val_dice)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "trained_models/transformer_weighted_best.pth")
        patience_counter = 0
        print(">> New best (val loss). Model saved!")
    elif val_dice_score > best_val_dice:
        best_val_dice = val_dice_score
        torch.save(model.state_dict(), "trained_models/transformer_weighted_best.pth")
        patience_counter = 0
        print(">> New best (val dice). Model saved!")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(">> Early stopping triggered!")
            break

# End
torch.save(model.state_dict(), "trained_models/transformer_weighted_last.pth")
