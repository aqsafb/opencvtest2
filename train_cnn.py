# train_cnn.py
import torch, torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

tf = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ColorJitter(0.3, 0.3, 0.3, 0.05),
    transforms.ToTensor(),
])
ds = datasets.ImageFolder("cls_data", transform=tf)   # kelas: background=0, buoy=1
dl = DataLoader(ds, batch_size=32, shuffle=True)

model = nn.Sequential(
    nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 32x32
    nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 16x16
    nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 8x8
    nn.Flatten(), nn.Linear(64 * 8 * 8, 64), nn.ReLU(),
    nn.Dropout(0.3), nn.Linear(64, 2),
)

opt = torch.optim.Adam(model.parameters(), 1e-3)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(20):
    total = 0
    for x, y in dl:
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
        total += loss.item()
    print(f"epoch {epoch+1} loss {total/len(dl):.4f}")
torch.save(model.state_dict(), "buoy_cnn.pth")

model.eval()
torch.onnx.export(model, torch.randn(1, 3, 64, 64), "buoy_cnn.onnx",
                  input_names=["input"], output_names=["output"])
print(ds.class_to_idx)   # cek urutan kelas