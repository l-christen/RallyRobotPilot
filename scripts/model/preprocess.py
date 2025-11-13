import os
import lzma
import pickle
import torch
from tqdm import tqdm
import numpy as np

def scale_image(img):
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # (H, W, C) -> (C, H, W)
    return img

def scale_speed(speed):
    return min(speed, 50.0) / 50.0

def scale_raycasts(raycasts):
    raycasts = np.array(raycasts, dtype=np.float32)
    raycasts = np.clip(raycasts, 0, 100.0) / 100.0
    return raycasts

# --- CONFIG ---
RAW_DIR = "data"             # dossier avec record_*.npz
OUT_DIR = "preprocessed"     # dossier de sortie .pt
SEQ_LEN = 15
SKIP = 2

os.makedirs(OUT_DIR, exist_ok=True)

def process_record(file_path):
    """Découpe un record_x.npz en séquences torch.tensor"""
    with lzma.open(file_path, "rb") as f:
        data = pickle.load(f)

    sequences = []
    for i in range(SKIP, len(data) - SEQ_LEN - 2):  # -2 car cible = t+2
        seq = data[i:i+SEQ_LEN]
        images, raycasts, speeds = [], [], []

        for msg in seq:
            img = msg.image
            img = scale_image(img)
            images.append(torch.tensor(img, dtype=torch.float32))
            raycasts.append(scale_raycasts(msg.raycast_distances))
            speeds.append(scale_speed(msg.car_speed))

        # cible = commandes à t+2 frames
        target_controls = data[i + SEQ_LEN + 1].current_controls
        target_controls = torch.tensor(target_controls, dtype=torch.float32)

        # empilement final
        images = torch.stack(images)  # (seq_len, C, H, W)
        raycasts = torch.tensor(raycasts[-1], dtype=torch.float32)
        speed = torch.tensor(speeds[-1], dtype=torch.float32)

        sequences.append((images, raycasts, speed, target_controls))

    return sequences


def main():
    all_files = [f for f in os.listdir(RAW_DIR) if f.endswith(".npz")]
    total_sequences = 0

    for fname in tqdm(all_files, desc="Préprocessing"):
        fpath = os.path.join(RAW_DIR, fname)
        try:
            seqs = process_record(fpath)
            for j, sample in enumerate(seqs):
                torch.save(sample, os.path.join(OUT_DIR, f"{fname[:-4]}_{j:05d}.pt"))
            total_sequences += len(seqs)
        except Exception as e:
            print(f"[X] Erreur sur {fname}: {e}")

    print(f"[✓] {total_sequences} séquences enregistrées dans {OUT_DIR}")

if __name__ == "__main__":
    main()
