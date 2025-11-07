import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from tqdm import tqdm
import os
from model import CNNLSTMModel
import pickle
import lzma

import os
import pickle
import lzma
import torch
from torch.utils.data import Dataset

class VideoGameDataset(Dataset):
    """
    Dataset PyTorch pour les données RallyRobotPilot.
    Chaque élément correspond à une séquence temporelle
    (15 frames, raycasts, vitesse, directions).
    """

    def __init__(self, data_path, transform=None, seq_len=15, skip=2):
        """
        Args:
            data_path (str): dossier contenant les .npz
            transform (callable): transformations d'image (resize, normalize, etc.)
            seq_len (int): taille des séquences temporelles
            skip (int): nombre d’images initiales à ignorer (ex: 2)
        """
        self.data_path = data_path
        self.transform = transform
        self.seq_len = seq_len
        self.skip = skip

        self.samples = self.compile_all_data()

    def open_data_file(self, file_path):
        """Ouvre un fichier .npz contenant des SensingSnapshot"""
        with lzma.open(file_path, "rb") as file:
            data = pickle.load(file)
        if not isinstance(data, list):
            raise ValueError(f"Unexpected format in {file_path}")
        return data

    def process_data(self, data):
        """Découpe les données en séquences de longueur seq_len"""
        processed = []
        for i in range(self.skip, len(data) - self.seq_len + 1):
            seq = data[i:i+self.seq_len]

            images, raycasts, speeds, directions = [], [], [], []

            for msg in seq:
                img = msg.image
                if self.transform:
                    img = self.transform(img)
                images.append(img)

                raycasts.append(msg.raycast_distances)
                speeds.append(msg.car_speed)
                directions.append(msg.current_controls)

            images = torch.stack(images)  # (seq_len, C, H, W)
            raycasts = torch.tensor(raycasts[-1], dtype=torch.float32)  # dernière frame
            speed = torch.tensor(speeds[-1], dtype=torch.float32)       # dernière frame
            directions = torch.tensor(directions[-1], dtype=torch.float32)  # dernière frame

            processed.append((images, raycasts, speed, directions))

        return processed

    def compile_all_data(self):
        """Charge toutes les runs et compile toutes les séquences"""
        all_sequences = []
        for file_name in os.listdir(self.data_path):
            if file_name.endswith(".npz"):
                file_path = os.path.join(self.data_path, file_name)
                try:
                    data = self.open_data_file(file_path)
                    seqs = self.process_data(data)
                    all_sequences.extend(seqs)
                except Exception as e:
                    print(f"[X] Erreur sur {file_name}: {e}")
        print(f"[✓] {len(all_sequences)} séquences compilées depuis {self.data_path}")
        return all_sequences
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]



class MultiTaskLoss(nn.Module):
    """
    Loss multi-tâches avec pondération automatique (uncertainty weighting)
    """
    def __init__(self):
        super(MultiTaskLoss, self).__init__()
        # Paramètres apprenables pour la pondération
        self.log_var_raycast = nn.Parameter(torch.zeros(1))
        self.log_var_speed = nn.Parameter(torch.zeros(1))
        self.log_var_classification = nn.Parameter(torch.zeros(1))
        
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
    
    def forward(self, pred_raycasts, pred_speed, pred_classification, 
                target_raycasts, target_speed, target_classification):
        # Loss pour raycasts (régression)
        loss_raycast = self.mse(pred_raycasts, target_raycasts)
        
        # Loss pour vitesse (régression)
        loss_speed = self.mse(pred_speed, target_speed)
        
        # Loss pour classification (multi-label)
        loss_classification = self.bce(pred_classification, target_classification)
        
        # Pondération automatique avec uncertainty
        precision_raycast = torch.exp(-self.log_var_raycast)
        precision_speed = torch.exp(-self.log_var_speed)
        precision_classification = torch.exp(-self.log_var_classification)
        
        total_loss = (precision_raycast * loss_raycast + self.log_var_raycast +
                      precision_speed * loss_speed + self.log_var_speed +
                      precision_classification * loss_classification + self.log_var_classification)
        
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
    BATCH_SIZE = 8
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-4
    # app = Ursina(size=(160, 224)), keep this image size
    IMG_HEIGHT = 224
    IMG_WIDTH = 160
    
    # Chemins
    DATA_PATH = "./data"
    CHECKPOINT_DIR = "./checkpoints"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Datasets
    dataset = VideoGameDataset(DATA_PATH)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
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
    model = CNNLSTMModel(img_height=IMG_HEIGHT, img_width=IMG_WIDTH).to(device)
    print(f"Nombre de paramètres: {model.get_num_parameters():,}")
    
    # Loss et optimizer
    criterion = MultiTaskLoss().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
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