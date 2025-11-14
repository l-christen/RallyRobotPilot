import os
import lzma
import pickle
import torch
from tqdm import tqdm
import numpy as np
import copy

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
SEQ_LEN = 40
SKIP = 2

os.makedirs(OUT_DIR, exist_ok=True)

import cv2

def add_realistic_noise(img):
    """
    img : (H, W, 3), float32 ou uint8
    Retour : img bruitée en float32 dans [0,1]
    """

    # Convertir en float32 si besoin
    if img.dtype != np.float32:
        img = img.astype(np.float32) / 255.0

    noisy = copy.deepcopy(img)

    # ---- 1. Petit shift gamma (simule variations d'exposition)
    gamma = 1.0 + np.random.uniform(-0.10, 0.10)
    noisy = np.clip(noisy ** gamma, 0, 1)

    # ---- 2. Petit offset RGB (simule lumière dynamique du jeu)
    offset = np.random.normal(0, 0.015, size=(1,1,3))
    noisy = np.clip(noisy + offset, 0, 1)

    # ---- 3. Gaussian blur très léger (motion blur / capture)
    if np.random.rand() < 0.3:
        noisy = cv2.GaussianBlur(noisy, (3,3), sigmaX=0.5)

    # ---- 4. Bruit corrélé spatialement (perceptuellement plus réaliste)
    if np.random.rand() < 0.5:
        noise = np.random.normal(0, 0.015, img.shape).astype(np.float32)
        # filtrer le bruit pour qu'il soit corrélé → plus réaliste
        noise = cv2.GaussianBlur(noise, (3,3), sigmaX=0.5)
        noisy = np.clip(noisy + noise, 0, 1)

    return noisy


def process_record(file_path):
    """Découpe un record_x.npz en séquences torch.tensor"""
    with lzma.open(file_path, "rb") as f:
        data = pickle.load(f)

    sequences = []
    for i in range(SKIP, len(data) - SEQ_LEN - 2):  # -2 car cible = t+2
        seq = data[i:i+SEQ_LEN]
        images, raycasts, speeds = [], [], []

        add_noise = (np.random.rand() < 0.5)
        noise_seed = np.random.randint(0, 1e9)

        for msg in seq:
            img = msg.image
            if add_noise:
                np.random.seed(noise_seed)
                img = add_realistic_noise(img)
                
            img = scale_image(img)
            images.append(torch.tensor(img, dtype=torch.float32))
            raycasts.append(scale_raycasts(msg.raycast_distances))
            speeds.append(scale_speed(msg.car_speed))

        # cible = commandes à t+2 frames
        target_controls = data[i + SEQ_LEN + 1].current_controls
        inverted_target_controls = copy.deepcopy(target_controls)
        fwd, back, left, right = inverted_target_controls
        inverted_target_controls = [fwd, back, right, left]  # inverser gauche/droite
        inverted_target_controls = torch.tensor(inverted_target_controls, dtype=torch.float32)
        target_controls = torch.tensor(target_controls, dtype=torch.float32)

        noise_seed = np.random.randint(0, 1e9)

        # invert sequences for data augmentation
        inverted_images, inverted_raycasts = [], []
        for msg in seq:
            img = msg.image[:, ::-1, :].copy()  # flip horizontal
            if add_noise:
                np.random.seed(noise_seed)
                img = add_realistic_noise(img)
            img = scale_image(img)
            inverted_images.append(torch.tensor(img, dtype=torch.float32))
            rc = scale_raycasts(msg.raycast_distances)[::-1].copy()  # flip raycasts
            inverted_raycasts.append(rc)
            
        

        # empilement final
        images = torch.stack(images)  # (seq_len, C, H, W)
        inverted_images = torch.stack(inverted_images)
        raycasts = torch.tensor(raycasts[-1], dtype=torch.float32)
        inverted_raycasts = torch.tensor(inverted_raycasts[-1], dtype=torch.float32)
        speed = torch.tensor(speeds[-1], dtype=torch.float32)

        sequences.append((images, raycasts, speed, target_controls))
        sequences.append((inverted_images, inverted_raycasts, speed, inverted_target_controls))

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
