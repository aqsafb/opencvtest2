# convert_dataset.py
import cv2, os, glob, random

DATASET = r"D:\ROV\buoy.v3i.yolov8"
OUT = "cls_data"
SPLITS = ["train", "valid"]      # test tidak dipakai
PAD = 0.15                       # tambah margin 15% di sekitar buoy
BG_PER_IMAGE = 2                 # jumlah crop background per gambar

os.makedirs(f"{OUT}/buoy", exist_ok=True)
os.makedirs(f"{OUT}/background", exist_ok=True)


def overlap(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return not (ax + aw <= bx or bx + bw <= ax or ay + ah <= by or by + bh <= ay)


def read_boxes(label_path, W, H):
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    for line in open(label_path):
        p = line.split()
        if len(p) == 5:                       # format kotak: cls cx cy w h
            cx, cy, w, h = [float(v) for v in p[1:]]
            x, y = (cx - w / 2) * W, (cy - h / 2) * H
            boxes.append((int(x), int(y), int(w * W), int(h * H)))
        elif len(p) > 5:                      # format polygon: ambil kotak pembungkus
            xs = [float(v) * W for v in p[1::2]]
            ys = [float(v) * H for v in p[2::2]]
            boxes.append((int(min(xs)), int(min(ys)),
                          int(max(xs) - min(xs)), int(max(ys) - min(ys))))
    return boxes


n_buoy = n_bg = 0
for split in SPLITS:
    pattern = os.path.join(DATASET, split, "images", "*.*")
    for img_path in glob.glob(pattern):
        img = cv2.imread(img_path)
        if img is None:
            continue
        H, W = img.shape[:2]
        name = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(DATASET, split, "labels", name + ".txt")
        boxes = read_boxes(label_path, W, H)

        # crop buoy
        for i, (x, y, w, h) in enumerate(boxes):
            if w < 8 or h < 8:
                continue
            px, py = int(w * PAD), int(h * PAD)
            x0, y0 = max(0, x - px), max(0, y - py)
            x1, y1 = min(W, x + w + px), min(H, y + h + py)
            crop = img[y0:y1, x0:x1]
            if crop.size:
                cv2.imwrite(f"{OUT}/buoy/{split}_{name}_{i}.jpg", crop)
                n_buoy += 1

        # crop background acak yang tidak menimpa buoy
        for j in range(BG_PER_IMAGE):
            for _ in range(20):
                if boxes:
                    _, _, bw, bh = random.choice(boxes)
                    bw, bh = max(bw, 40), max(bh, 40)
                else:
                    bw = bh = random.randint(40, 120)
                if bw >= W or bh >= H:
                    continue
                bx = random.randint(0, W - bw)
                by = random.randint(0, H - bh)
                if any(overlap((bx, by, bw, bh), b) for b in boxes):
                    continue
                crop = img[by:by + bh, bx:bx + bw]
                cv2.imwrite(f"{OUT}/background/{split}_{name}_bg{j}.jpg", crop)
                n_bg += 1
                break

print(f"buoy: {n_buoy} gambar, background: {n_bg} gambar")