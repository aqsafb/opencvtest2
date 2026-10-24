# train_yolo.py
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.train(
    data=r"D:/ROV/buoy.v3i.yolov8/data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    patience=20,          # berhenti sendiri kalau tidak membaik
    hsv_h=0.02, hsv_s=0.7, hsv_v=0.5,   # augmentasi warna/cahaya (kondisi air berubah-ubah)
    degrees=10,           # rotasi kecil
    fliplr=0.5,
    mosaic=1.0,
)