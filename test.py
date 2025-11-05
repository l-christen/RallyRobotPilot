import lzma
import pickle

with lzma.open("data/record_0.npz", "rb") as f:
    data = pickle.load(f)

print(len(data))            # nombre de snapshots
print(type(data[0]))        # type de message

import cv2
import numpy as np

img = data[0].image
if img is not None:
    cv2.imshow("frame", img)
    cv2.waitKey(0)

import os
os.makedirs("extracted_imgs", exist_ok=True)

for i, msg in enumerate(data):
    if msg.image is None:
        continue

    img = cv2.cvtColor(msg.image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f"extracted_imgs/frame_{i:05}.jpg", img)