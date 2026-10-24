# train_cnn_loc.py
import os, glob, random
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models

DATASET = r"D:/ROV/buoy.v3i.yolov8"
SIZE = 224
EPOCHS = 30
BATCH = 32
dev = "cuda" if torch.cuda.is_available() else "cpu"


def read_label(path):
    """Ambil kotak buoy terbesar dari file label (format YOLO box atau polygon)."""
    best, box = 0, None
    if not os.path.exists(path):
        return None
    for line in open(path):
        v = line.split()
        if len(v) == 5:
            cx, cy, w, h = map(float, v[1:])
        elif len(v) > 5:
            xs = [float(a) for a in v[1::2]]
            ys = [float(a) for a in v[2::2]]
            w, h = max(xs) - min(xs), max(ys) - min(ys)
            cx, cy = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2
        else:
            continue
        if w * h > best:
            best, box = w * h, (cx, cy, w, h)
    return box


def load_items(split):
    items = []
    for p in glob.glob(os.path.join(DATASET, split, "images", "*.*")):
        name = os.path.splitext(os.path.basename(p))[0]
        box = read_label(os.path.join(DATASET, split, "labels", name + ".txt"))
        items.append((p, box))
    return items


class BuoyDS(Dataset):
    def __init__(self, items, train):
        self.items, self.train = items, train

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, box = self.items[i]
        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (SIZE, SIZE))
        t = [0.0, 0.0, 0.0, 0.0, 0.0] if box is None else [1.0, *box]

        if self.train:
            if random.random() < 0.5:                       # flip horizontal
                img = img[:, ::-1]
                if t[0] == 1.0:
                    t[1] = 1.0 - t[1]
            img = img.astype(np.float32)
            img = img * random.uniform(0.6, 1.4) + random.uniform(-30, 30)   # terang/gelap
            img = img * np.array([random.uniform(0.85, 1.15) for _ in range(3)],
                                 dtype=np.float32)                           # pergeseran warna
            img = np.clip(img, 0, 255)

        img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
        x = torch.from_numpy(img).permute(2, 0, 1)
        return x, torch.tensor(t, dtype=torch.float32)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        self.features = m.features                         # keluaran 1280 x 7 x 7
        self.reduce = nn.Sequential(nn.Conv2d(1280, 64, 1), nn.ReLU())
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 5),                             # obj, cx, cy, w, h
        )
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):                                  # x: RGB, nilai 0-1
        x = (x - self.mean) / self.std
        return self.head(self.reduce(self.features(x)))


def loss_fn(out, t):
    obj = t[:, 0]
    l_obj = F.binary_cross_entropy_with_logits(out[:, 0], obj)
    pos = obj > 0.5
    if pos.any():
        pred = torch.sigmoid(out[pos, 1:5])
        l_box = F.smooth_l1_loss(pred, t[pos, 1:5]) * 5
    else:
        l_box = torch.zeros((), device=out.device)
    return l_obj + l_box


def iou(a, b):
    ax1, ay1, ax2, ay2 = a[:, 0] - a[:, 2] / 2, a[:, 1] - a[:, 3] / 2, a[:, 0] + a[:, 2] / 2, a[:, 1] + a[:, 3] / 2
    bx1, by1, bx2, by2 = b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2, b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2
    iw = (torch.min(ax2, bx2) - torch.max(ax1, bx1)).clamp(min=0)
    ih = (torch.min(ay2, by2) - torch.max(ay1, by1)).clamp(min=0)
    inter = iw * ih
    union = a[:, 2] * a[:, 3] + b[:, 2] * b[:, 3] - inter
    return inter / union.clamp(min=1e-6)


def main():
    train_items, val_items = load_items("train"), load_items("valid")
    print(f"train: {len(train_items)} gambar, valid: {len(val_items)} gambar, device: {dev}")

    tl = DataLoader(BuoyDS(train_items, True), batch_size=BATCH, shuffle=True, num_workers=0)
    vl = DataLoader(BuoyDS(val_items, False), batch_size=BATCH, shuffle=False, num_workers=0)

    model = Net().to(dev)
    opt = torch.optim.Adam([
        {"params": model.features.parameters(), "lr": 1e-4},
        {"params": list(model.reduce.parameters()) + list(model.head.parameters()), "lr": 1e-3},
    ])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)

    best = 1e9
    for ep in range(EPOCHS):
        model.train()
        tr = 0
        for x, t in tl:
            x, t = x.to(dev), t.to(dev)
            opt.zero_grad()
            loss = loss_fn(model(x), t)
            loss.backward()
            opt.step()
            tr += loss.item()
        sched.step()

        model.eval()
        vloss, correct, n, ious = 0, 0, 0, []
        with torch.no_grad():
            for x, t in vl:
                x, t = x.to(dev), t.to(dev)
                out = model(x)
                vloss += loss_fn(out, t).item()
                pred_obj = (torch.sigmoid(out[:, 0]) > 0.5).float()
                correct += (pred_obj == t[:, 0]).sum().item()
                n += len(t)
                pos = t[:, 0] > 0.5
                if pos.any():
                    ious.append(iou(torch.sigmoid(out[pos, 1:5]), t[pos, 1:5]).cpu())
        vloss /= len(vl)
        miou = torch.cat(ious).mean().item() if ious else 0
        print(f"epoch {ep+1}/{EPOCHS} train {tr/len(tl):.4f} val {vloss:.4f} "
              f"acc_ada_buoy {correct/n:.3f} IoU_rata2 {miou:.3f}")

        if vloss < best:
            best = vloss
            torch.save(model.state_dict(), "buoy_loc.pth")

    model.load_state_dict(torch.load("buoy_loc.pth", map_location=dev))
    model.eval().cpu()
    torch.onnx.export(model, torch.randn(1, 3, SIZE, SIZE), "buoy_loc.onnx",
                      input_names=["input"], output_names=["output"])
    print("selesai: buoy_loc.onnx dan buoy_loc.pth tersimpan")


if __name__ == "__main__":
    main()