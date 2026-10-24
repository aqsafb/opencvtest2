# capture_negatives.py
import cv2, os, time

OUT = "negatives"
INTERVAL = 0.4          # detik antar foto
os.makedirs(OUT, exist_ok=True)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)

n = len(os.listdir(OUT))
last = 0
print("Gerakkan kamera ke berbagai arah (wajah, dinding, meja, lantai). Tekan q untuk berhenti.")
while True:
    ok, frame = cap.read()
    if not ok:
        break
    if time.time() - last > INTERVAL:
        cv2.imwrite(f"{OUT}/neg_{n:04d}.jpg", frame)
        n += 1
        last = time.time()
    cv2.putText(frame, f"tersimpan: {n}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow("capture negatives", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()