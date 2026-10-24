# vision_cnn.py
import cv2
import numpy as np

net = cv2.dnn.readNetFromONNX("buoy_cnn.onnx")

# Range HSV LONGGAR untuk cari kandidat. Sesuaikan dengan warna buoy kamu.
HSV_LOW = np.array([0, 67, 66], dtype=np.uint8)
HSV_HIGH = np.array([179, 248, 240], dtype=np.uint8)
MIN_AREA = 300
CNN_CONF = 0.8


def proposals(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, HSV_LOW, HSV_HIGH)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.dilate(mask, np.ones((7, 7), np.uint8))
    cv2.imshow("mask", mask)
    cv2.waitKey(1)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [cv2.boundingRect(c) for c in cnts if cv2.contourArea(c) > MIN_AREA]


def is_buoy(crop):
    blob = cv2.dnn.blobFromImage(crop, 1 / 255.0, (64, 64), swapRB=True)
    net.setInput(blob)
    logits = net.forward()[0]
    p = np.exp(logits - logits.max())
    p /= p.sum()
    return float(p[1])            # index 1 = buoy


def open_camera():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Kamera gagal dibuka")
        raise SystemExit
    return cap


def main():
    cap = open_camera()
    SHOW = True                   # set False kalau di robot tanpa layar

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        H, W = frame.shape[:2]
        best = None
        for (x, y, w, h) in proposals(frame):
            crop = frame[y:y + h, x:x + w]
            if crop.size == 0:
                continue
            conf = is_buoy(crop)
            if conf > CNN_CONF and (best is None or conf > best[0]):
                best = (conf, x, y, w, h)

        if best:
            conf, x, y, w, h = best
            error_x = ((x + w / 2) - W / 2) / (W / 2)
            print(f"buoy conf={conf:.2f} error_x={error_x:+.2f} area={(w*h)/(W*H):.3f}")
            if SHOW:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"buoy {conf:.2f}", (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            print("no detection")

        if SHOW:
            cv2.imshow("vision", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()