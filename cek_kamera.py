import cv2

for backend_name, backend in [("DSHOW", cv2.CAP_DSHOW), ("MSMF", cv2.CAP_MSMF), ("DEFAULT", cv2.CAP_ANY)]:
    for idx in range(4):
        cap = cv2.VideoCapture(idx, backend)
        ok, _ = cap.read() if cap.isOpened() else (False, None)
        print(f"{backend_name} index {idx}: {'BERHASIL' if ok else 'gagal'}")
        cap.release()