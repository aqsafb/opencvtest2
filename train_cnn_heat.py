# train_cnn_heat.py
import os, glob, random
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models

DATASET = r"D:/ROV/buoy.v3i.yolov8"
NEG_DIR = "negatives"
SIZE = 320
STRIDE = 8
G = SIZE // STRIDE            # grid 40 x 40
EPOCHS = 25
BATCH = 16
dev = "cuda" if torch.cuda.is_available() else "cpu"


def read_boxes(path):
    boxes = []
    if not os.path.exists(path):
        return boxes
    for line in open(path):
        v = line.split()
        if len(v) == 5:
            cx, cy, w, h = map(float, v[1:])
        elif len(v) > 5:                                  # format polygon
            xs = [float(a) for a in v[1::2]]
            ys = [float(a) for a in v[2::2]]
            w, h = max(xs) - min(xs), max(ys) - min(ys)
            cx, cy = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2
        else:
            continue
        boxes.append((cx, cy, min(w, 1.0), min(h, 1.0)))
    return boxes


def load_items(split):
    items = []
    for p in glob.glob(os.path.join(DATASET, split, "images", "*.*")):
        name = os.path.splitext(os.path.basename(p))[0]
        items.append((p, read_boxes(os.path.join(DATASET, split, "labels", name + ".txt"))))
    return items


def load_negatives():
    return [(p, []) for p in glob.glob(os.path.join(NEG_DIR, "*.*"))]


class BuoyDS(Dataset):
    def __init__(self, items, train):
        self.items, self.train = items, train

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, boxes = self.items[i]
        img = cv2.imread(path)
        if img is None:
            return self[(i + 1) % len(self.items)]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (SIZE, SIZE))
        boxes = [list(b) for b in boxes]

        if self.train:
            if random.random() < 0.5:                      # flip horizontal
                img = np.ascontiguousarray(img[:, ::-1])
                for b in boxes:
                    b[0] = 1.0 - b[0]

            # zoom in/out acak: buoy jadi kecil (jauh) atau besar (dekat)
            s = random.uniform(0.5, 1.5)
            lo, hi = sorted((0.0, SIZE * (1 - s)))
            tx, ty = random.uniform(lo, hi), random.uniform(lo, hi)
            M = np.float32([[s, 0, tx], [0, s, ty]])
            img = cv2.warpAffine(img, M, (SIZE, SIZE), borderValue=(114, 114, 114))
            nb = []
            for cx, cy, w, h in boxes:
                ncx, ncy = (cx * SIZE * s + tx) / SIZE, (cy * SIZE * s + ty) / SIZE
                if 0 <= ncx < 1 and 0 <= ncy < 1:
                    nb.append([ncx, ncy, min(w * s, 1.0), min(h * s, 1.0)])
            boxes = nb

            img = img.astype(np.float32)
            img = img * random.uniform(0.6, 1.4) + random.uniform(-30, 30)       # terang/gelap
            img = img * np.array([random.uniform(0.85, 1.15) for _ in range(3)], dtype=np.float32)
            tint = np.array([random.uniform(0, 60), random.uniform(80, 160), random.uniform(80, 160)],
                            dtype=np.float32)
            a = random.uniform(0, 0.35)                                            # kabut/tint air
            img = img * (1 - a) + tint * a
            if random.random() < 0.3:
                img = cv2.GaussianBlur(img, (5, 5), 0)
            img = np.clip(img, 0, 255)

        img = img.astype(np.float32) / 255.0
        x = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1)

        heat = np.zeros((1, G, G), np.float32)
        wh = np.zeros((2, G, G), np.float32)
        mask = np.zeros((1, G, G), np.float32)
        ys, xs = np.mgrid[0:G, 0:G]
        for cx, cy, w, h in boxes:
            ci, cj = min(G - 1, int(cx * G)), min(G - 1, int(cy * G))
            sigma = max(0.8, min(w, h) * G / 3)
            g = np.exp(-((xs - ci) ** 2 + (ys - cj) ** 2) / (2 * sigma ** 2))
            heat[0] = np.maximum(heat[0], g)
            heat[0, cj, ci] = 1.0
            wh[:, cj, ci] = (w, h)
            mask[0, cj, ci] = 1.0
        return x, torch.from_numpy(heat), torch.from_numpy(wh), torch.from_numpy(mask)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        f = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT).features
        self.low = f[:7]          # stride 8, 32 channel (detail kecil)
        self.high = f[7:14]       # stride 16, 96 channel (makna/konteks)
        self.fuse = nn.Sequential(
            nn.Conv2d(32 + 96, 96, 3, padding=1), nn.BatchNorm2d(96), nn.ReLU(),
            nn.Conv2d(96, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
        )
        self.out = nn.Conv2d(64, 3, 1)             # heat, w, h
        nn.init.constant_(self.out.bias[0], -2.19)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):                          # x: RGB 0-1
        x = (x - self.mean) / self.std
        a = self.low(x)
        b = F.interpolate(self.high(a), scale_factor=2.0, mode="nearest")
        return self.out(self.fuse(torch.cat([a, b], 1)))


def focal(logit, gt):
    p = torch.sigmoid(logit).clamp(1e-4, 1 - 1e-4)
    pos = gt.eq(1).float()
    pos_l = -torch.log(p) * (1 - p) ** 2 * pos
    neg_l = -torch.log(1 - p) * p ** 2 * (1 - gt) ** 4 * (1 - pos)
    return (pos_l.sum() + neg_l.sum()) / pos.sum().clamp(min=1)


def loss_fn(out, heat, wh, mask):
    l_heat = focal(out[:, 0:1], heat)
    pred_wh = torch.sigmoid(out[:, 1:3])
    l_wh = (torch.abs(pred_wh - wh) * mask).sum() / (mask.sum() * 2).clamp(min=1)
    return l_heat + 10 * l_wh


def main():
    train_items, val_items = load_items("train"), load_items("valid")
    negs = load_negatives()
    random.shuffle(negs)
    k = int(len(negs) * 0.8)
    train_items += negs[:k]
    val_items += negs[k:]
    if not negs:
        print("PERINGATAN: folder 'negatives' kosong/tidak ada. Jalankan capture_negatives.py dulu!")
    print(f"train: {len(train_items)}, valid: {len(val_items)}, negatif: {len(negs)}, device: {dev}")

    tl = DataLoader(BuoyDS(train_items, True), batch_size=BATCH, shuffle=True, num_workers=0)
    vl = DataLoader(BuoyDS(val_items, False), batch_size=BATCH, shuffle=False, num_workers=0)

    model = Net().to(dev)
    backbone = list(model.low.parameters()) + list(model.high.parameters())
    newparams = list(model.fuse.parameters()) + list(model.out.parameters())
    opt = torch.optim.Adam([{"params": backbone, "lr": 1e-4}, {"params": newparams, "lr": 1e-3}])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)

    best = -9.0
    for ep in range(EPOCHS):
        model.train()
        tr = 0.0
        for x, heat, wh, mask in tl:
            x, heat, wh, mask = x.to(dev), heat.to(dev), wh.to(dev), mask.to(dev)
            opt.zero_grad()
            loss = loss_fn(model(x), heat, wh, mask)
            loss.backward()
            opt.step()
            tr += loss.item()
        sched.step()

        model.eval()
        tp = npos = fp = nneg = 0
        with torch.no_grad():
            for x, heat, wh, mask in vl:
                x, heat = x.to(dev), heat.to(dev)
                p = torch.sigmoid(model(x)[:, 0])                   # B,G,G
                mx, idx = p.flatten(1).max(1)
                gt = heat[:, 0].flatten(1)
                has = gt.max(1).values >= 1
                gt_at = gt.gather(1, idx[:, None])[:, 0]
                hit = (mx > 0.5) & (gt_at > 0.2)
                tp += (hit & has).sum().item()
                npos += has.sum().item()
                fp += ((mx > 0.5) & ~has).sum().item()
                nneg += (~has).sum().item()
        recall = tp / max(npos, 1)
        fp_rate = fp / max(nneg, 1)
        print(f"epoch {ep+1}/{EPOCHS} train {tr/len(tl):.4f} | "
              f"recall {recall:.3f} (buoy ketemu & posisi benar) | "
              f"false_positive {fp_rate:.3f} ({fp}/{nneg} gambar tanpa buoy)")

        score = recall - fp_rate
        if score > best:
            best = score
            torch.save(model.state_dict(), "buoy_heat.pth")

    model.load_state_dict(torch.load("buoy_heat.pth", map_location=dev))
    model.eval().cpu()
    torch.onnx.export(model, torch.randn(1, 3, SIZE, SIZE), "buoy_heat.onnx",
                      input_names=["input"], output_names=["output"])
    print("selesai: buoy_heat.onnx dan buoy_heat.pth tersimpan")


if __name__ == "__main__":
    main()