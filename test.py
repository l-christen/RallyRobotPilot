import lzma
import pickle

with lzma.open("data/record_0.npz", "rb") as f:
    data = pickle.load(f)

print(len(data))            # nombre de snapshots
print(type(data[0]))        # type de message

import cv2
import numpy as np
last_idx = len(data) - 1
img = data[last_idx].image
print(img.shape, img.min(), img.max())
if img is not None:
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imshow("Frame", img_bgr)
    cv2.waitKey(0)

times = []
for i, msg in enumerate(data):
    times.append(msg.timestamp)

times = np.array(times)
intervals = times[1:] - times[:-1]
print("Average interval: ", np.mean(intervals))
print("Min interval: ", np.min(intervals))
print("Max interval: ", np.max(intervals))
# distribution histogram
import matplotlib.pyplot as plt
plt.hist(intervals, bins=20)
plt.title("Histogram of time intervals between snapshots")
plt.xlabel("Interval (s)")
plt.ylabel("Frequency")
plt.show()