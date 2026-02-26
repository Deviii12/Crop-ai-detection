import time, copy
from pathlib import Path
from contextlib import nullcontext

import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import f1_score

DATA = Path("data/split")
OUT  = Path("models"); OUT.mkdir(exist_ok=True)

IMG_SIZE = 224
BATCH    = 32
EPOCHS   = 60
LR       = 3e-4
WD       = 1e-4
PATIENCE = 10
WARMUP_EPOCHS = 5

# Device
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"
print(f"✅ Using device: {DEVICE}")

# Data transforms
mean = [0.485, 0.456, 0.406]; std = [0.229, 0.224, 0.225]
t_train = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(25),
    transforms.ColorJitter(0.3,0.3,0.3,0.05),
    transforms.RandomPerspective(distortion_scale=0.4, p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])
t_eval = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

# Datasets
dtr = datasets.ImageFolder(DATA/"train", transform=t_train)
dvl = datasets.ImageFolder(DATA/"val",   transform=t_eval)
dte = datasets.ImageFolder(DATA/"test",  transform=t_eval)

# ✅ Use num_workers=0 for Mac compatibility
tr = DataLoader(dtr, batch_size=BATCH, shuffle=True,  num_workers=0)
vl = DataLoader(dvl, batch_size=BATCH, shuffle=False, num_workers=0)
te = DataLoader(dte, batch_size=BATCH, shuffle=False, num_workers=0)

classes = dtr.classes
num_classes = len(classes)
print(f"🗂 Classes: {num_classes}")

# Class weights
targets = torch.tensor(dtr.targets)
counts  = torch.bincount(targets, minlength=num_classes).float()
weights = (counts.sum() / counts).clamp(min=1.0)
weights = weights / weights.mean()
class_weights = weights.to(DEVICE)

# Model
from torchvision.models import resnet50, ResNet50_Weights
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, num_classes)
model.to(DEVICE)

# Freeze backbone initially
for p in model.parameters(): p.requires_grad = False
for p in model.fc.parameters(): p.requires_grad = True

# Optimizer / scheduler / loss
opt = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=WD)
sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3)
criterion = nn.CrossEntropyLoss(weight=class_weights)

# AMP
use_autocast = DEVICE in ("cuda", "mps")
if DEVICE == "cuda":
    scaler = torch.cuda.amp.GradScaler()
else:
    scaler = None
amp_ctx = (torch.autocast(device_type=DEVICE, dtype=torch.float16) if use_autocast else nullcontext())

def evaluate():
    model.eval()
    ce_sum, y_true, y_pred = 0.0, [], []
    with torch.no_grad():
        for x,y in vl:
            x,y = x.to(DEVICE), y.to(DEVICE)
            with amp_ctx:
                logits = model(x)
                loss = criterion(logits,y)
            ce_sum += loss.item()*x.size(0)
            y_true += y.cpu().tolist()
            y_pred += logits.argmax(1).cpu().tolist()
    f1 = f1_score(y_true,y_pred,average="macro")
    return ce_sum/len(dvl), f1

def evaluate_test():
    model.eval()
    ce_sum, y_true, y_pred = 0.0, [], []
    with torch.no_grad():
        for x,y in te:
            x,y = x.to(DEVICE), y.to(DEVICE)
            with amp_ctx:
                logits = model(x); loss = criterion(logits,y)
            ce_sum += loss.item()*x.size(0)
            y_true += y.cpu().tolist()
            y_pred += logits.argmax(1).cpu().tolist()
    f1 = f1_score(y_true,y_pred,average="macro")
    return ce_sum/len(dte), f1, (y_true,y_pred)

# Training
best = {"f1":0.0, "state":None, "epoch":0}
no_improve = 0
t0 = time.time()

for epoch in range(1, EPOCHS+1):
    epoch_start = time.time()  # ⏱ per-epoch timer

    if epoch == WARMUP_EPOCHS+1:
        for p in model.parameters(): p.requires_grad = True
        opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
        print("🔓 Unfroze backbone for fine-tuning.")

    model.train()
    run_loss = 0.0
    for x,y in tr:
        x,y = x.to(DEVICE), y.to(DEVICE)
        opt.zero_grad(set_to_none=True)
        if DEVICE == "cuda":
            with torch.cuda.amp.autocast():
                logits = model(x)
                loss = criterion(logits,y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            with amp_ctx:
                logits = model(x)
                loss = criterion(logits,y)
            loss.backward()
            opt.step()
        run_loss += loss.item()*x.size(0)

    v_loss, v_f1 = evaluate()
    sched.step(v_f1)

    # timing info
    epoch_time = time.time() - epoch_start
    total_time = time.time() - t0
    eta = (EPOCHS - epoch) * (epoch_time / 60)

    # print LR manually
    for param_group in opt.param_groups:
        current_lr = param_group['lr']
    print(f"Epoch {epoch:02d} | train_loss={(run_loss/len(dtr)):.4f}  "
          f"val_loss={v_loss:.4f}  val_f1={v_f1:.4f}  "
          f"LR={current_lr:.6f}  time={epoch_time:.1f}s  total={total_time/60:.1f}m  ETA={eta:.1f}m")

    if v_f1 > best["f1"]:
        best = {"f1":v_f1, "state":copy.deepcopy(model.state_dict()), "epoch":epoch}
        no_improve = 0
    else:
        no_improve += 1
        if no_improve >= PATIENCE:
            print("⏹ Early stopping.")
            break

# Save best
stamp = int(time.time())
ckpt_path = OUT / f"best_resnet50_{stamp}.pt"
torch.save({"state_dict":best["state"], "classes":classes, "arch":"resnet50"}, ckpt_path)
print(f"✅ Saved best model from epoch {best['epoch']} (val_f1={best['f1']:.4f}) -> {ckpt_path}")

# Test
model.load_state_dict(best["state"])
t_loss, t_f1, _ = evaluate_test()
print(f"📊 TEST loss={t_loss:.4f}  TEST F1={t_f1:.4f}  (elapsed={(time.time()-t0)/60:.1f} min)")
print("🎉 Training complete.")