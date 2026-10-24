# vision_cnn_heat.py
import cv2
import numpy as np

net = cv2.dnn.readNetFromONNX("buoy_heat.onnx")
SIZE = 320
CONF = 0.5
CONFIRM_FRAMES = 3
LOST_FRAMES = 5
ALPHA = 0.5
SHOW = True


def sigmoid(v):
    return 1.0 / (1.0 + np.exp(-v))


def detect(frame):
    """Kembalikan daftar (skor, cx, cy, w, h) relatif 0-1, diurutkan dari skor tertinggi."""
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (SIZE, SIZE), swapRB=True)
    net.setInput(blob)
    out = net.forward()[0]                         # (3, G, G)
    heat = sigmoid(out[0]).astype(np.float32)
    wh = sigmoid(out[1:3])
    G = heat.shape[0]
    peak = (heat == cv2.dilate(heat, np.ones((3, 3), np.uint8))) & (heat > CONF)
    ys, xs = np.nonzero(peak)
    dets = [(float(heat[y, x]), (x + 0.5) / G, (y + 0.5) / G,
             float(wh[0, y, x]), float(wh[1, y, x])) for y, x in zip(ys, xs)]
    dets.sort(reverse=True)
    return dets, float(heat.max())


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
    seen = lost = 0
    smooth = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        H, W = frame.shape[:2]

        dets, top = detect(frame)
        if dets:
            conf, cx, cy, w, h = dets[0]
            seen += 1
            lost = 0
            cur = np.array([cx, cy, w, h])
            smooth = cur if smooth is None else ALPHA * cur + (1 - ALPHA) * smooth
        else:
            lost += 1
            if lost > LOST_FRAMES:
                seen, smooth = 0, None

        if smooth is not None and seen >= CONFIRM_FRAMES:
            sx, sy, sw, sh = smooth
            error_x = (sx - 0.5) / 0.5
            print(f"buoy conf={dets[0][0] if dets else 0:.2f} error_x={error_x:+.2f} area={sw*sh:.3f}")
            if SHOW:
                x1, y1 = int((sx - sw / 2) * W), int((sy - sh / 2) * H)
                x2, y2 = int((sx + sw / 2) * W), int((sy + sh / 2) * H)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, "buoy", (x1, max(15, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            print(f"no detection (puncak tertinggi={top:.2f})")

        if SHOW:
            cv2.imshow("vision", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()