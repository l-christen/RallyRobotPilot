import os
import lzma
import pickle
import torch
from tqdm import tqdm
import numpy as np
from collections import defaultdict
from get_distrib import get_distrib


# ======================
# CONFIG
# ======================
RAW_DIR = "data"
OUT_DIR = "preprocessed"

SEQ_LEN = 40
SKIP = 2

# max par classe (d'après ta distrib : 2266)
MAX_PER_CLASS = 2266

# classes incohérentes qu'on ignore
INCOHERENT = {
    (1, 1, 0, 0),
    (0, 0, 1, 1),
    (1, 1, 1, 1),
    (1, 0, 1, 1),
    (0, 1, 1, 1),
}

os.makedirs(OUT_DIR, exist_ok=True)


# ======================
# HELPERS
# ======================
def scale_image(img):
    """
    img : torch.Tensor ou np.ndarray, shape (C,H,W)
    - Si img est float32 -> on suppose valeurs 0–255, on divise par 255.
    - Si img est uint8   -> convertit en float32 puis /255
    Retour : float32 dans [0,1]
    """
    
    # Si numpy
    if isinstance(img, np.ndarray):
        if img.dtype == np.float32:
            return img / 255.0
        elif img.dtype == np.uint8:
            return img.astype(np.float32) / 255.0
        else:
            raise ValueError(f"Unsupported dtype: {img.dtype}")

    # Si tensor PyTorch
    if isinstance(img, torch.Tensor):
        if img.dtype == torch.float32:
            return img / 255.0
        elif img.dtype == torch.uint8:
            return img.float() / 255.0
        else:
            raise ValueError(f"Unsupported tensor dtype: {img.dtype}")

    raise TypeError("img must be NumPy array or torch.Tensor")

def tuple_from_controls(ctrl):
    """Convertit une liste/tensor de contrôles en tuple python hashable"""
    return (int(ctrl[0]), int(ctrl[1]), int(ctrl[2]), int(ctrl[3]))


def flip_controls(ctrl):
    """Inversion left/right pour les labels"""
    fw, bw, le, ri = ctrl
    return (fw, bw, ri, le)


def should_accept(combo, kept):
    """Équilibrage progressif basé sur un quota MAX_PER_CLASS"""
    if combo in INCOHERENT:
        return False

    current = kept[combo]

    if current >= MAX_PER_CLASS:
        return False

    # ratio progression
    ratio = current / MAX_PER_CLASS

    # probabilité décroissante
    p_accept = 1.0 - ratio  # linéaire, stable et simple

    return np.random.rand() < p_accept


# ======================
# MAIN PROCESSING
# ======================
def process_record(path, kept):
    """Découpage d'un record_x.npz en séquences équilibrées"""

    with lzma.open(path, "rb") as f:
        data = pickle.load(f)

    sequences = []

    N = len(data)
    for i in range(SKIP, N - SEQ_LEN - 2):

        # ---------------------
        # LABEL (cible = t+1)
        # ---------------------
        raw_ctrl = data[i + SEQ_LEN].current_controls
        combo = tuple_from_controls(raw_ctrl)

        # équilibrage
        if not should_accept(combo, kept):
            continue

        kept[combo] += 1

        # ---------------------
        # Séquence d'images
        # ---------------------
        seq = data[i:i+SEQ_LEN]

        images = []
        raycasts = []
        speeds = []

        for msg in seq:
            # STOCKAGE RGB EN UINT8 ***
            img = msg.image.astype(np.uint8)
            img = np.transpose(img, (2,0,1))  # C,H,W
            images.append(torch.from_numpy(img))  # uint8

            rc = np.array(msg.raycast_distances, dtype=np.float32)
            rc = np.clip(rc, 0, 100.0)
            raycasts.append(rc)

            speed = min(msg.car_speed, 50.0)
            speeds.append(speed / 50.0)

        images = torch.stack(images)  # (T,3,H,W) uint8

        # on prend le dernier raycast et speed
        raycasts = torch.tensor(raycasts[-1] / 100.0, dtype=torch.float32)
        speed = torch.tensor(speeds[-1], dtype=torch.float32)

        # label tensor
        target_controls = torch.tensor(combo, dtype=torch.float32)


        # ---------------------
        # FLIP VERSION
        # ---------------------
        flip_imgs = []

        for msg in seq:
            inv = msg.image[:, ::-1, :].copy().astype(np.uint8)
            inv = np.transpose(inv, (2,0,1))
            flip_imgs.append(torch.from_numpy(inv))

        flip_imgs = torch.stack(flip_imgs)

        # raycasts inversés
        flip_raycasts = torch.tensor(
            raycasts.numpy()[::-1].copy(),
            dtype=torch.float32
        )

        # labels inversés left <-> right
        flip_combo = flip_controls(combo)
        flip_controls_tensor = torch.tensor(flip_combo, dtype=torch.float32)

        kept[flip_combo] += 1

        # empilement final
        sequences.append((images, raycasts, speed, target_controls))
        sequences.append((flip_imgs, flip_raycasts, speed, flip_controls_tensor))

    return sequences


# ======================
# MAIN
# ======================
def main():
    print("[+] Calcul distribution initiale…")
    global_distrib = get_distrib()

    # compteur des séquences retenues par classe
    kept = defaultdict(int)

    files = [f for f in os.listdir(RAW_DIR) if f.endswith(".npz")]
    total = 0

    print("[+] Début preprocess…")

    for fname in tqdm(files):
        path = os.path.join(RAW_DIR, fname)

        try:
            seqs = process_record(path, kept)
            for j, s in enumerate(seqs):
                torch.save(s, os.path.join(OUT_DIR, f"{fname[:-4]}_{j:05d}.pt"))
            total += len(seqs)
        except Exception as e:
            print(f"[X] Erreur {fname}: {e}")

    print("\n========== DONE ==========")
    print(f"[✓] Total séquences enregistrées : {total}")
    print("[✓] Sequences par classe (post-undersampling) :")
    for c, k in kept.items():
        print(f" {c} : {k} (max={MAX_PER_CLASS})")


if __name__ == "__main__":
    main()
