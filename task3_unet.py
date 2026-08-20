"""
task3_unet.py
Task 3 - U-Net segmentation in PyTorch: dataset, model, losses, Dice/IoU
metrics, training loop, side-by-side prediction panels, and curve plots.

Setup:
    pip install torch torchvision

Run:
    python task3_unet.py
Adjust IMG_DIR / MASK_DIR to your dataset. Image i must correspond to mask i
after sorting - check this or Dice will stay near zero.

NOTE: The assignment provides a U-Net skeleton on Canvas. If they test against
its specific class structure, replace the UNet class below but keep the rest.
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

IMG_DIR = "dataset/images"    # <-- EDIT
MASK_DIR = "dataset/masks"    # <-- EDIT


# ---------- Dataset ----------
class SegDataset(Dataset):
    """Pairs each image with its mask by sorted filename order."""

    def __init__(self, img_dir, mask_dir, size=(256, 256)):
        self.imgs = sorted(glob.glob(os.path.join(img_dir, "*")))
        self.masks = sorted(glob.glob(os.path.join(mask_dir, "*")))
        assert len(self.imgs) == len(self.masks), "image/mask count mismatch"
        self.size = size

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        img = Image.open(self.imgs[i]).convert("L").resize(self.size)
        mask = Image.open(self.masks[i]).convert("L").resize(self.size)
        img = np.array(img, dtype=np.float32) / 255.0
        mask = (np.array(mask, dtype=np.float32) > 127).astype(np.float32)
        return (torch.from_numpy(img).unsqueeze(0),
                torch.from_numpy(mask).unsqueeze(0))


# ---------- Model ----------
def double_conv(ci, co):
    return nn.Sequential(
        nn.Conv2d(ci, co, 3, padding=1), nn.BatchNorm2d(co), nn.ReLU(inplace=True),
        nn.Conv2d(co, co, 3, padding=1), nn.BatchNorm2d(co), nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    """Small 3-level U-Net: single-channel input, single-channel mask output."""

    def __init__(self):
        super().__init__()
        self.d1 = double_conv(1, 32)
        self.d2 = double_conv(32, 64)
        self.d3 = double_conv(64, 128)
        self.pool = nn.MaxPool2d(2)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.u2 = double_conv(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.u1 = double_conv(64, 32)
        self.out = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        c1 = self.d1(x)
        c2 = self.d2(self.pool(c1))
        c3 = self.d3(self.pool(c2))                       # bottleneck
        u2 = self.u2(torch.cat([self.up2(c3), c2], 1))
        u1 = self.u1(torch.cat([self.up1(u2), c1], 1))
        return self.out(u1)                               # logits (no sigmoid)


# ---------- Metrics ----------
def dice_iou(logits, target, eps=1e-6):
    """Mean Dice and IoU over a batch. Applies sigmoid + 0.5 threshold."""
    prob = torch.sigmoid(logits)
    pred = (prob > 0.5).float()
    inter = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2 * inter + eps) / (union + eps)
    iou = (inter + eps) / (union - inter + eps)
    return dice.mean().item(), iou.mean().item()


# ---------- Losses (BCE / Dice / BCE+Dice for the loss-ablation extra credit) ----------
def dice_loss(logits, target, eps=1e-6):
    prob = torch.sigmoid(logits)
    inter = (prob * target).sum(dim=(1, 2, 3))
    union = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return (1 - (2 * inter + eps) / (union + eps)).mean()


def bce_loss(logits, target):
    return F.binary_cross_entropy_with_logits(logits, target)


def bce_dice(logits, target):
    return bce_loss(logits, target) + dice_loss(logits, target)


# ---------- Training ----------
def train_unet(train_ds, val_ds, epochs=20, lr=1e-3, loss_fn=bce_dice, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    tl = DataLoader(train_ds, batch_size=4, shuffle=True)
    vl = DataLoader(val_ds, batch_size=4)

    history = {"loss": [], "val_dice": [], "val_iou": []}
    for ep in range(epochs):
        model.train()
        run = 0.0
        for x, y in tl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            run += loss.item()

        model.eval()
        ds, is_ = [], []
        with torch.no_grad():
            for x, y in vl:
                x, y = x.to(device), y.to(device)
                d, i = dice_iou(model(x), y)
                ds.append(d)
                is_.append(i)

        history["loss"].append(run / len(tl))
        history["val_dice"].append(float(np.mean(ds)))
        history["val_iou"].append(float(np.mean(is_)))
        print(f"epoch {ep+1}: loss {history['loss'][-1]:.4f} "
              f"dice {history['val_dice'][-1]:.4f} iou {history['val_iou'][-1]:.4f}")
    return model, history


# ---------- Visualisation ----------
def show_predictions(model, val_ds, n=3, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    n = min(n, len(val_ds))
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
    if n == 1:
        axes = axes[None, :]
    for r in range(n):
        x, y = val_ds[r]
        with torch.no_grad():
            pred = torch.sigmoid(model(x.unsqueeze(0).to(device)))[0, 0].cpu().numpy()
        axes[r, 0].imshow(x[0], cmap="gray"); axes[r, 0].set_title("input")
        axes[r, 1].imshow(y[0], cmap="gray"); axes[r, 1].set_title("ground truth")
        axes[r, 2].imshow(pred > 0.5, cmap="gray"); axes[r, 2].set_title("prediction")
        for c in range(3):
            axes[r, c].axis("off")
    plt.tight_layout()
    plt.savefig("unet_predictions.png", dpi=150)
    plt.show()


def plot_curves(history):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(history["loss"])
    ax[0].set_title("Training loss")
    ax[0].set_xlabel("epoch")
    ax[1].plot(history["val_dice"], label="Dice")
    ax[1].plot(history["val_iou"], label="IoU")
    ax[1].set_title("Validation metrics")
    ax[1].set_xlabel("epoch")
    ax[1].legend()
    plt.tight_layout()
    plt.savefig("unet_curves.png", dpi=150)
    plt.show()


def make_splits(img_dir=IMG_DIR, mask_dir=MASK_DIR, val_frac=0.2, seed=0):
    full = SegDataset(img_dir, mask_dir)
    n_val = max(1, int(len(full) * val_frac))
    return torch.utils.data.random_split(
        full, [len(full) - n_val, n_val],
        generator=torch.Generator().manual_seed(seed))


if __name__ == "__main__":
    train_ds, val_ds = make_splits()
    model, history = train_unet(train_ds, val_ds, epochs=20)
    plot_curves(history)
    show_predictions(model, val_ds, n=3)
    torch.save(model.state_dict(), "unet.pt")
    print("saved model to unet.pt")
