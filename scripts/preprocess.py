import os
import lzma
import pickle
import torch
import numpy as np
from tqdm import tqdm


# ======================
# CONFIG
# ======================
RAW_DIR = "data"
OUT_DIR = "preprocessed"

SEQ_LEN = 4     # nombre de frames passées au CNN
SKIP = 2        # pour éviter les frames initiales foireuses

os.makedirs(OUT_DIR, exist_ok=True)


# ======================
# HELPERS
# ======================
def to_uint8_chw(img):
    """Convertit image Panda3D RGB -> uint8 C,H,W"""
    img = img.astype(np.uint8)
    return np.transpose(img, (2, 0, 1))  # C,H,W


def flip_img_uint8(img_3hw):
    """Flip horizontal d’un tensor uint8 (C,H,W)"""
    return img_3hw[:, :, ::-1].copy()


def clip_and_norm_raycast(values):
    """Normalise raycasts distances dans [0,1]"""
    arr = np.array(values, dtype=np.float32)
    arr = np.clip(arr, 0, 100.0)
    return arr / 100.0


def norm_speed(v):
    """Normalise speed dans [0,1], cap à 50 km/h"""
    v = min(v, 50.0)
    return np.float32(v / 50.0)


def tuple_ctrl(ctrl):
    """(fw,bw,left,right) -> tuple d’int"""
    return tuple(int(v) for v in ctrl)


def weighted_mean_controls(data, idxs):
    """Soft label (moyenne pondérée t-3 → t+1)"""
    weights = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    ctrls = []

    for i in idxs:
        ctrls.append(np.array(tuple_ctrl(data[i].current_controls), dtype=np.float32))

    ctrls = np.stack(ctrls)  # (4,4)
    return np.average(ctrls, axis=0, weights=weights)  # (4,)


# ======================
# PROCESS 1 RECORD
# ======================
def process_record(path, save_prefix):
    """
    Renvoie une liste de séquences prêtes à être sauvées :
    - images: (T,3,H,W) uint8
    - raycasts: (15,)
    - speed: scalar
    - soft_controls: (4,)
    """

    with lzma.open(path, "rb") as f:
        data = pickle.load(f)

    out_items = []
    N = len(data)

    # pour chaque séquence glissante
    for i in range(SKIP, N - SEQ_LEN):

        # Indices pour le label t-3 → t+1
        label_idxs = [
            i + SEQ_LEN - 3,
            i + SEQ_LEN - 2,
            i + SEQ_LEN - 1,
            i + SEQ_LEN
        ]

        # Séquence de frames
        seq = data[i : i + SEQ_LEN]

        # =====================
        # IMAGES (T,3,H,W)
        # =====================
        imgs_uint8 = []
        for msg in seq:
            img_chw = to_uint8_chw(msg.image)
            imgs_uint8.append(torch.from_numpy(img_chw))

        images = torch.stack(imgs_uint8)  # (T,3,H,W) uint8

        # =====================
        # RAYCAST & SPEED (sur la dernière frame)
        # =====================
        last_msg = seq[-1]

        ray = clip_and_norm_raycast(last_msg.raycast_distances)
        speed = norm_speed(last_msg.car_speed)

        # =====================
        # SOFT LABEL (4,)
        # =====================
        soft_ctrl = weighted_mean_controls(data, label_idxs)
        soft_ctrl_tensor = torch.tensor(soft_ctrl, dtype=torch.float32)

        # =====================
        # VERSION FLIPPÉE
        # =====================
        flipped_imgs = []
        for msg in seq:
            img = to_uint8_chw(msg.image)
            flipped_imgs.append(torch.from_numpy(flip_img_uint8(img)))
        flipped = torch.stack(flipped_imgs)

        # flip des classes : swap left/right
        flip_soft = soft_ctrl.copy()
        flip_soft[2], flip_soft[3] = flip_soft[3], flip_soft[2]
        flip_soft_tensor = torch.tensor(flip_soft, dtype=torch.float32)

        # flip raycasts (si ordonnés gauche→droite)
        flip_ray = ray[::-1].copy()

        # =====================
        # STOCKAGE (pas de RAM qui explose)
        # =====================
        out_items.append((images, torch.tensor(ray), torch.tensor(speed), soft_ctrl_tensor))
        out_items.append((flipped, torch.tensor(flip_ray), torch.tensor(speed), flip_soft_tensor))

    return out_items


# ======================
# MAIN
# ======================
def main():
    files = [f for f in os.listdir(RAW_DIR) if f.endswith(".npz")]

    print(f"[+] Found {len(files)} raw demo files")

    counter = 0

    for fname in tqdm(files):
        path = os.path.join(RAW_DIR, fname)
        try:
            sequences = process_record(path, fname[:-4])
        except Exception as e:
            print(f"[ERROR] {fname}: {e}")
            continue

        # SAVE EACH ITEM AS SEPARATE .PT (low RAM)
        for j, item in enumerate(sequences):
            torch.save(item, os.path.join(OUT_DIR, f"{fname[:-4]}_{j:05d}.pt"))
            counter += 1

    print("\n===== DONE =====")
    print(f"[✓] Saved sequences: {counter}")


if __name__ == "__main__":
    main()