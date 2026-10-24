# vision_cnn_loc.py
import cv2
import numpy as np

net = cv2.dnn.readNetFromONNX("buoy_loc.onnx")
SIZE = 224
CONF = 0.6               # ambang yakin ada buoy
CONFIRM_FRAMES = 3       # harus terlihat 3 frame berturut-turut
LOST_FRAMES = 5          # toleransi hilang 5 frame
ALPHA = 0.5              # penghalusan posisi
SHOW = True


def sigmoid(v):
    return 1.0 / (1.0 + np.exp(-v))


def detect(frame):
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (SIZE, SIZE), swapRB=True)
    net.setInput(blob)
    out = net.forward()[0]
    conf = float(sigmoid(out[0]))
    cx, cy, w, h = sigmoid(out[1:5])
    return conf, cx, cy, w, h            # nilai 0-1 (relatif terhadap frame)


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

        conf, cx, cy, w, h = detect(frame)
        if conf > CONF:
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
            print(f"buoy conf={conf:.2f} error_x={error_x:+.2f} area={sw*sh:.3f}")
            if SHOW:
                x1, y1 = int((sx - sw / 2) * W), int((sy - sh / 2) * H)
                x2, y2 = int((sx + sw / 2) * W), int((sy + sh / 2) * H)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"buoy {conf:.2f}", (x1, max(15, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            print(f"no detection (conf={conf:.2f})")

        if SHOW:
            cv2.imshow("vision", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
    