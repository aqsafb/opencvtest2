# cari_hsv.py
import cv2, glob, numpy as np

files = glob.glob("cls_data/buoy/*.jpg")[:500]
pixels = []
for f in files:
    img = cv2.imread(f)
    if img is None:
        continue
    img = cv2.resize(img, (32, 32))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    hsv = hsv[(hsv[:, 1] > 60) & (hsv[:, 2] > 60)]   # buang piksel pucat/gelap (air, bayangan)
    pixels.append(hsv)

px = np.vstack(pixels)
hist = np.bincount(px[:, 0], minlength=180)
print("Hue paling sering muncul:", np.argsort(hist)[::-1][:5])

lo = np.percentile(px, 5, axis=0).astype(int)
hi = np.percentile(px, 95, axis=0).astype(int)
print("HSV_LOW  =", lo.tolist())
print("HSV_HIGH =", hi.tolist())