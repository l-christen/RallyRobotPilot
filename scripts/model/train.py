import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from tqdm import tqdm
import os
from model import ResNetLiteLSTM
import pickle
import lzma
from collections import defaultdict

def build_datasets(preproc_dir, train_ratio=0.8):
    # Regroupement par record
    record_groups = defaultdict(list)

    for f in sorted(os.listdir(preproc_dir)):
        if f.endswith(".pt"):
            record_name = f.split("_")[0] + "_" + f.split("_")[1]
            record_groups[record_name].append(os.path.join(preproc_dir, f))

    # Liste des records
    records = list(record_groups.keys())
    records.sort()

    # Split par record
    n = len(records)
    split = int(train_ratio * n)

    train_records = records[:split]
    val_records   = records[split:]

    # Construire les file lists
    train_files = []
    val_files   = []

    for r in train_records:
        train_files.extend(record_groups[r])
    for r in val_records:
        val_files.extend(record_groups[r])

    print("Records train :", train_records)
    print("Records val   :", val_records)
    print(f"Train files: {len(train_files)}")
    print(f"Val files:   {len(val_files)}")

    return VideoGameDataset(train_files), VideoGameDataset(val_files)

class VideoGameDataset(Dataset):
    def __init__(self, file_list, transform=None):
        self.files = file_list
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        images, raycasts, speed, directions = torch.load(self.files[idx])
        if self.transform:
            images = self.transform(images)
        return images, raycasts, speed, directions


class MultiTaskLoss(nn.Module):
    """
    Loss multi-tâches :
    - Raycasts      : régression MSE
    - Speed         : régression MSE
    - Classification: multi-label BCE

    → Uncertainty weighting (Kendall & Gal 2018)
    → + Boost manuel de la classification (commandes)
    """

    def __init__(self, command_weight=30.0):
        super().__init__()

        # Pondération automatique (apprise)
        self.log_var_raycast = nn.Parameter(torch.zeros(1))
        self.log_var_speed = nn.Parameter(torch.zeros(1))
        self.log_var_classification = nn.Parameter(torch.zeros(1))

        # Losses de base
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()

        # Pondération manuelle des commandes
        self.command_weight = command_weight


    def forward(self, pred_raycasts, pred_speed, pred_classification,
                target_raycasts, target_speed, target_classification):

        # --- Losses brutes ---
        loss_raycast = self.mse(pred_raycasts, target_raycasts)
        loss_speed = self.mse(pred_speed, target_speed)

        # classification boostée
        loss_classification = self.command_weight * self.bce(
            pred_classification, target_classification
        )

        # --- Uncertainty weighting ---
        precision_raycast = torch.exp(-self.log_var_raycast)
        precision_speed = torch.exp(-self.log_var_speed)
        precision_classification = torch.exp(-self.log_var_classification)

        total_loss = (
            precision_raycast * loss_raycast + self.log_var_raycast +
            precision_speed * loss_speed + self.log_var_speed +
            precision_classification * loss_classification + self.log_var_classification
        )

        return total_loss, loss_raycast, loss_speed, loss_classification


def train_epoch(model, dataloader, criterion, optimizer, scaler, device):
    """Une époque d'entraînement"""
    model.train()
    total_loss = 0
    total_raycast_loss = 0
    total_speed_loss = 0
    total_classification_loss = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for images, raycasts, speed, classification in pbar:
        images = images.to(device)
        raycasts = raycasts.to(device)
        speed = speed.to(device)
        classification = classification.to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision training
        with autocast():
            pred_raycasts, pred_speed, pred_classification = model(images)
            loss, loss_r, loss_s, loss_c = criterion(
                pred_raycasts, pred_speed, pred_classification,
                raycasts, speed, classification
            )
        
        # Backward avec gradient scaling
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        total_raycast_loss += loss_r.item()
        total_speed_loss += loss_s.item()
        total_classification_loss += loss_c.item()
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'raycast': f'{loss_r.item():.4f}',
            'speed': f'{loss_s.item():.4f}',
            'class': f'{loss_c.item():.4f}'
        })
    
    n = len(dataloader)
    return total_loss / n, total_raycast_loss / n, total_speed_loss / n, total_classification_loss / n


def validate(model, dataloader, criterion, device):
    """Validation"""
    model.eval()
    total_loss = 0
    total_raycast_loss = 0
    total_speed_loss = 0
    total_classification_loss = 0
    
    with torch.no_grad():
        for images, raycasts, speed, classification in tqdm(dataloader, desc="Validation"):
            images = images.to(device)
            raycasts = raycasts.to(device)
            speed = speed.to(device)
            classification = classification.to(device)
            
            with autocast():
                pred_raycasts, pred_speed, pred_classification = model(images)
                loss, loss_r, loss_s, loss_c = criterion(
                    pred_raycasts, pred_speed, pred_classification,
                    raycasts, speed, classification
                )
            
            total_loss += loss.item()
            total_raycast_loss += loss_r.item()
            total_speed_loss += loss_s.item()
            total_classification_loss += loss_c.item()
    
    n = len(dataloader)
    return total_loss / n, total_raycast_loss / n, total_speed_loss / n, total_classification_loss / n


def main():
    # Hyperparamètres
    BATCH_SIZE = 128
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-4
    # app = Ursina(size=(160, 224)), keep this image size
    IMG_HEIGHT = 224
    IMG_WIDTH = 160
    
    # Chemins
    DATA_PATH = "preprocessed"
    CHECKPOINT_DIR = "./checkpoints"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Datasets
    train_dataset, val_dataset = build_datasets(DATA_PATH, train_ratio=0.8)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    # Modèle
    model = ResNetLiteLSTM(lstm_hidden=256, lstm_layers=1).to(device)
    print(f"Nombre de paramètres: {model.get_num_parameters():,}")
    
    # Loss et optimizer
    criterion = MultiTaskLoss(command_weight=30.0).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Gradient scaler pour mixed precision
    scaler = GradScaler()
    
    # Entraînement
    best_val_loss = float('inf')
    
    for epoch in range(NUM_EPOCHS):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        print(f"{'='*50}")
        
        # Train
        train_loss, train_r, train_s, train_c = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device
        )
        
        # Validation
        val_loss, val_r, val_s, val_c = validate(
            model, val_loader, criterion, device
        )
        
        # Scheduler step
        scheduler.step(val_loss)
        
        # Logs
        print(f"\nTrain Loss: {train_loss:.4f} (R:{train_r:.4f}, S:{train_s:.4f}, C:{train_c:.4f})")
        print(f"Val Loss: {val_loss:.4f} (R:{val_r:.4f}, S:{val_s:.4f}, C:{val_c:.4f})")
        
        # Sauvegarder le meilleur modèle
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }
            torch.save(checkpoint, os.path.join(CHECKPOINT_DIR, 'best_model.pth'))
            print(f"✓ Meilleur modèle sauvegardé (val_loss: {val_loss:.4f})")
        
        # Sauvegarder checkpoint périodique
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }
            torch.save(checkpoint, os.path.join(CHECKPOINT_DIR, f'checkpoint_epoch_{epoch+1}.pth'))
    
    print("\n✓ Entraînement terminé!")


if __name__ == "__main__":
    main()