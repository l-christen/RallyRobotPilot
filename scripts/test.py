import os
import lzma
import pickle
from collections import Counter

DATA_DIR = "data"   # Dossier contenant les record_x.npz

# -------------------------------------------
# Chargement de tous les fichiers .npz
# -------------------------------------------
files = sorted(
    [f for f in os.listdir(DATA_DIR) if f.endswith(".npz")]
)

if not files:
    print("[X] Aucun fichier .npz trouvé dans data/")
    exit()

print(f"[+] Fichiers trouvés : {files}")

global_counter = Counter()
total_frames = 0
loaded_files = 0

# -------------------------------------------
# Lecture + comptage global
# -------------------------------------------
for fname in files:
    path = os.path.join(DATA_DIR, fname)

    try:
        with lzma.open(path, "rb") as f:
            data = pickle.load(f)

        loaded_files += 1
        print(f"[+] Chargé {len(data)} frames depuis {fname}")

        for msg in data:
            fw, bw, le, ri = msg.current_controls
            combo = (int(fw), int(bw), int(le), int(ri))
            global_counter[combo] += 1

        total_frames += len(data)

    except Exception as e:
        print(f"[!] Erreur en chargeant {fname} : {e}")
        continue

# -------------------------------------------
# Affichage des résultats
# -------------------------------------------
print("\n==============================")
print("Combinaisons de touches (globales)")
print("(fw, back, left, right) : count")
print("==============================\n")

for combo, count in global_counter.most_common():
    print(f"{combo} : {count}")

print("\nTotal frames :", total_frames)
print("Total fichiers chargés :", loaded_files)

# Proportions
print("\n=== Proportions globales (%) ===")
for combo, count in global_counter.most_common():
    pct = (count / total_frames) * 100
    print(f"{combo} : {pct:.2f}%")
