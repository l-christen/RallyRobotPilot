import os
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import numpy as np
import ast

from model import StackedResNetDriving


# ============================
# Utils images
# ============================
def scale_image_tensor(img):
    if img.dtype == torch.uint8:
        return img.float() / 255.0
    elif img.dtype == torch.float32:
        return img / 255.0
    else:
        raise ValueError(f"Unsupported image dtype: {img.dtype}")


# ============================
# Dataset
# ============================
class DrivingDataset(Dataset):
    def __init__(self, file_list):
        self.files = file_list

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        images, ray, speed, actions = torch.load(self.files[idx])
        images = scale_image_tensor(images)
        return images, ray.float(), speed.float(), actions.float()


# ============================
# Split train / val
# ============================
def build_datasets(preproc_dir, train_ratio=0.8):

    record_groups = defaultdict(list)

    for f in sorted(os.listdir(preproc_dir)):
        if not f.endswith(".pt"):
            continue
        parts = f.split("_")
        if len(parts) < 3:
            key = "all"
        else:
            key = parts[0] + "_" + parts[1]
        record_groups[key].append(os.path.join(preproc_dir, f))

    records = sorted(record_groups.keys())
    split = int(train_ratio * len(records))

    train_files = []
    val_files = []

    for r in records[:split]:
        train_files.extend(record_groups[r])
    for r in records[split:]:
        val_files.extend(record_groups[r])

    print(f"[INFO] #records train: {len(records[:split])}")
    print(f"[INFO] #records val:   {len(records[split:])}")
    print(f"[INFO] #samples train: {len(train_files)}")
    print(f"[INFO] #samples val:   {len(val_files)}")

    return DrivingDataset(train_files), DrivingDataset(val_files)


# ======================================================
# NEW : COMBO-BASED WEIGHTS (not per-touch)
# ======================================================
def compute_combo_weights(
    path="preprocessed/global_distrib.txt",
    eps=1e-9
):
    """
    Lit un fichier de distribution sous la forme :
        (1, 0, 0, 0) : 18740
        (0, 0, 0, 1) : 9728
        ...
    
    Retourne :
        dict combo -> weight normalisé (moyenne = 1)
    """

    if not os.path.exists(path):
        raise FileNotFoundError(f"Distribution file not found: {path}")

    distrib = {}

    # ---------------------------
    # Lecture du fichier ligne par ligne
    # ---------------------------
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Exemple de ligne :
            # "(1, 0, 0, 0) : 18740"
            try:
                combo_str, count_str = line.split(":")
                combo = ast.literal_eval(combo_str.strip())
                count = int(count_str.strip())
                distrib[combo] = count
            except Exception as e:
                print(f"[WARN] Could not parse line: {line}  ({e})")

    if len(distrib) == 0:
        raise RuntimeError(f"No valid data found in {path}")

    total = sum(distrib.values())

    # ---------------------------
    # Fréquence par combo
    # ---------------------------
    freq = {combo: count / total for combo, count in distrib.items()}

    # ---------------------------
    # Poids = inverse de la fréquence
    # ---------------------------
    weights = {
        combo: 1.0 / max(f, eps)
        for combo, f in freq.items()
    }

    # ---------------------------
    # Normalisation → moyenne = 1
    # ---------------------------
    mean_w = np.mean(list(weights.values()))
    for combo in weights:
        weights[combo] /= mean_w

    # ---------------------------
    # Log
    # ---------------------------
    print("[INFO] Combo weights loaded from file:")
    for combo, w in weights.items():
        print(f"  {combo} : {w:.4f}")

    return weights


# ============================
# Multi-task loss with combo weights
# ============================
class MultiTaskLoss(nn.Module):
    def __init__(self,
                 combo_weights,
                 w_action=1.0,
                 w_raycast=0.05,
                 w_speed=0.05):
        super().__init__()

        self.bce = nn.BCEWithLogitsLoss(reduction="none")  # we weight manually
        self.mse = nn.MSELoss()

        self.combo_weights = combo_weights
        self.w_action = w_action
        self.w_raycast = w_raycast
        self.w_speed = w_speed

    def forward(self,
                pred_ray, pred_speed, pred_actions,
                target_ray, target_speed, target_actions):

        # ---------
        # Action loss (BCE)
        # ---------
        bce_raw = self.bce(pred_actions, target_actions)  # (B,4)
        bce_mean_per_sample = bce_raw.mean(dim=1)         # (B,)

        # Determine combo for each sample
        rounded = torch.round(target_actions).cpu().numpy()
        weights = []

        for r in rounded:
            tup = tuple(int(x) for x in r)
            if tup in self.combo_weights:
                weights.append(self.combo_weights[tup])
            else:
                # if unseen combo: neutral weight
                weights.append(1.0)

        weights = torch.tensor(weights, dtype=torch.float32, device=pred_actions.device)

        loss_action = (bce_mean_per_sample * weights).mean()

        # ---------
        # Aux tasks
        # ---------
        loss_ray = self.mse(pred_ray, target_ray)
        loss_speed = self.mse(pred_speed, target_speed)

        loss = (
            self.w_action * loss_action +
            self.w_raycast * loss_ray +
            self.w_speed * loss_speed
        )

        return loss, loss_ray, loss_speed, loss_action


# ============================
# Train / Val
# ============================
def train_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    total_loss = total_r = total_s = total_a = 0.0
    pbar = tqdm(loader, desc="Train")

    for images, ray, speed, actions in pbar:
        images = images.to(device)
        ray = ray.to(device)
        speed = speed.to(device).unsqueeze(1)
        actions = actions.to(device)

        optimizer.zero_grad()

        with autocast():
            pred_ray, pred_speed, pred_actions = model(images)
            loss, lr, ls, la = criterion(
                pred_ray, pred_speed, pred_actions,
                ray, speed, actions
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        total_r += lr.item()
        total_s += ls.item()
        total_a += la.item()

    n = len(loader)
    return total_loss/n, total_r/n, total_s/n, total_a/n


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = total_r = total_s = total_a = 0.0

    with torch.no_grad():
        pbar = tqdm(loader, desc="Val")
        for images, ray, speed, actions in pbar:
            images = images.to(device)
            ray = ray.to(device)
            speed = speed.to(device).unsqueeze(1)
            actions = actions.to(device)

            with autocast():
                pred_ray, pred_speed, pred_actions = model(images)
                loss, lr, ls, la = criterion(
                    pred_ray, pred_speed, pred_actions,
                    ray, speed, actions
                )

            total_loss += loss.item()
            total_r += lr.item()
            total_s += ls.item()
            total_a += la.item()

    n = len(loader)
    return total_loss/n, total_r/n, total_s/n, total_a/n


# ============================
# MAIN
# ============================
def main():

    DATA_DIR = "preprocessed"
    CHECKPOINT_DIR = "checkpoints"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    BATCH_SIZE = 300
    NUM_EPOCHS = 50
    LR = 1e-4
    TRAIN_RATIO = 0.8

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] Device:", device)

    # datasets
    train_ds, val_ds = build_datasets(DATA_DIR, TRAIN_RATIO)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=4, pin_memory=True)

    # --- Combo weights ---
    combo_weights = compute_combo_weights()

    # model
    model = StackedResNetDriving(num_frames=2).to(device)
    print(f"[INFO] #params: {model.get_num_parameters():,}")

    # loss
    criterion = MultiTaskLoss(combo_weights).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )

    scaler = GradScaler()

    best_val = float("inf")

    for epoch in range(NUM_EPOCHS):
        print("\n" + "="*60)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")

        train_loss, train_r, train_s, train_a = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device
        )
        val_loss, val_r, val_s, val_a = validate(
            model, val_loader, criterion, device
        )

        scheduler.step(val_loss)

        print(f"[TRAIN] loss={train_loss:.4f}")
        print(f"[VAL]   loss={val_loss:.4f}")

        # save best
        if val_loss < best_val:
            best_val = val_loss
            path = os.path.join(CHECKPOINT_DIR, "best_model.pth")
            torch.save({
                "epoch": epoch,
                "val_loss": val_loss,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }, path)
            print(f"[✓] Saved best model → {path}")

    print("\n[✓] Training finished.")


if __name__ == "__main__":
    main()
