import os, time, copy
from pathlib import Path
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import f1_score, classification_report
from sklearn.utils.multiclass import unique_labels
from src.models.custom_dwcnn import SmallDWNet

# ==========================
# Paths
# ==========================
DATA = Path("data/split")
OUT  = Path("models"); OUT.mkdir(exist_ok=True)

IMG_SIZE = 224
BATCH = 32       # Try 32, reduce to 16 if out-of-memory
EPOCHS = 25      # Increase if needed
LR = 3e-4

# Device selection (MPS for Apple Silicon, CUDA if NVIDIA GPU, else CPU)
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"
print(f"✅ Using device: {DEVICE}")

# ==========================
# Loaders
# ==========================
def get_loaders():
    mean = [0.485, 0.456, 0.406]; std = [0.229, 0.224, 0.225]
    t_train = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2,0.2,0.2,0.05),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    t_eval = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    dtr = datasets.ImageFolder(DATA/"train", transform=t_train)
    dvl = datasets.ImageFolder(DATA/"val", transform=t_eval)
    dte = datasets.ImageFolder(DATA/"test", transform=t_eval)

    return (
        DataLoader(dtr, batch_size=BATCH, shuffle=True),
        DataLoader(dvl, batch_size=BATCH, shuffle=False),
        DataLoader(dte, batch_size=BATCH, shuffle=False),
        dtr.classes
    )

# ==========================
# Evaluation
# ==========================
def evaluate(model, loader, device):
    model.eval()
    ce = nn.CrossEntropyLoss()
    loss_sum = 0; y_true=[]; y_pred=[]
    with torch.no_grad():
        for x,y in loader:
            x,y = x.to(device), y.to(device)
            logits = model(x)
            loss = ce(logits,y)
            loss_sum += loss.item()*x.size(0)
            y_true += y.cpu().tolist()
            y_pred += logits.argmax(1).cpu().tolist()
    f1 = f1_score(y_true,y_pred,average="macro")
    return loss_sum/len(loader.dataset), f1, (y_true,y_pred)

# ==========================
# Training Loop
# ==========================
if __name__ == "__main__":
    tr, vl, te, classes = get_loaders()
    model = SmallDWNet(num_classes=len(classes)).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=LR)
    ce = nn.CrossEntropyLoss()
    best = {"f1":0, "state":None}

    total_start = time.time()

    for epoch in range(1, EPOCHS+1):
        epoch_start = time.time()
        model.train(); run_loss=0

        for x,y in tr:
            x,y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            logits = model(x)
            loss = ce(logits,y)
            loss.backward(); opt.step()
            run_loss += loss.item()*x.size(0)

        v_loss, v_f1, _ = evaluate(model, vl, DEVICE)

        epoch_time = time.time() - epoch_start
        total_time = time.time() - total_start
        eta = (EPOCHS-epoch) * (epoch_time/60)

        print(f"Epoch {epoch:02d}: "
              f"train_loss={(run_loss/len(tr.dataset)):.4f}  "
              f"val_loss={v_loss:.4f}  "
              f"val_f1={v_f1:.4f}  "
              f"time={epoch_time:.1f}s  "
              f"total={total_time/60:.1f}min  "
              f"ETA={eta:.1f}min")

        if v_f1 > best["f1"]:
            best = {"f1":v_f1, "state":copy.deepcopy(model.state_dict())}

    # ==========================
    # Save best model
    # ==========================
    stamp = int(time.time())
    torch.save({"state_dict":best["state"], "classes":classes}, OUT/f"best_{stamp}.pt")

    # Final test evaluation
    model.load_state_dict(best["state"])
    t_loss, t_f1, (yt,yp) = evaluate(model, te, DEVICE)
    print("✅ Training complete. Best Val F1:", best["f1"])
    print("📊 TEST F1:", t_f1)

    # ✅ Safe classification report
    unique_idx = unique_labels(yt, yp)
    print(classification_report(
        yt, yp,
        labels=unique_idx,
        target_names=[classes[i] for i in unique_idx]
    ))

    print(f"⏱️ Total training time: {(time.time()-total_start)/60:.1f} minutes")
