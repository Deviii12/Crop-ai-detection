import time, copy, math, random
from pathlib import Path
from contextlib import nullcontext

import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
from sklearn.metrics import f1_score

# ==========================
# Paths & hyperparams
# ==========================
DATA = Path("data/split")
OUT  = Path("models"); OUT.mkdir(exist_ok=True)

IMG_SIZE = 224          
BATCH    = 32      
EPOCHS   = 30
LR_HEAD  = 3e-4       
LR_BODY  = 1e-4         
WD       = 1e-4
PATIENCE = 10          
WARMUP_EPOCHS = 5       

USE_MIXUP   = True
USE_CUTMIX  = True
MIXUP_ALPHA = 0.4
CUTMIX_ALPHA = 1.0
AUG_PROB    = 0.6      

# ==========================
# Device & AMP
# ==========================
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"
print(f"✅ Using device: {DEVICE}")

use_autocast = DEVICE in ("cuda", "mps")
amp_ctx = (torch.autocast(device_type=DEVICE, dtype=torch.float16) if use_autocast else nullcontext())
scaler = torch.cuda.amp.GradScaler() if DEVICE == "cuda" else None

# ==========================
# Data
# ==========================
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

dtr = datasets.ImageFolder(DATA/"train", transform=t_train)
dvl = datasets.ImageFolder(DATA/"val",   transform=t_eval)
dte = datasets.ImageFolder(DATA/"test",  transform=t_eval)

classes = dtr.classes
num_classes = len(classes)
print(f"🗂 Classes: {num_classes}")

# -------- Balanced sampling (no class weights to avoid double-counting)
targets = torch.tensor(dtr.targets)
counts  = torch.bincount(targets, minlength=num_classes).float()
class_inv = (1.0 / counts).clamp(max=10.0)  # cap extreme upsampling
sample_weights = class_inv[targets]
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

# ✅ Use num_workers=0 on macOS to avoid multiprocessing issues
tr = DataLoader(dtr, batch_size=BATCH, sampler=sampler, shuffle=False, num_workers=0, drop_last=True)
vl = DataLoader(dvl, batch_size=BATCH, shuffle=False, num_workers=0)
te = DataLoader(dte, batch_size=BATCH, shuffle=False, num_workers=0)

# ==========================
# Model (ResNet50 pretrained)
# ==========================
from torchvision.models import resnet50, ResNet50_Weights
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, num_classes)
model.to(DEVICE)

# Freeze all but FC head
for p in model.parameters(): p.requires_grad = False
for p in model.fc.parameters(): p.requires_grad = True

# stage 1 optimizer/scheduler (head only)
opt = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR_HEAD, weight_decay=WD)
sched = None  # we’ll start cosine after unfreezing

# Label smoothing helps on noisy/overlapping classes
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# ==========================
# MixUp / CutMix helpers
# ==========================
def mixup_batch(x, y, alpha=MIXUP_ALPHA):
    if alpha <= 0: return x, y, y, 1.0
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    idx = torch.randperm(x.size(0), device=x.device)
    x_mix = lam * x + (1 - lam) * x[idx]
    y_a, y_b = y, y[idx]
    return x_mix, y_a, y_b, lam

def rand_bbox(W, H, lam):
    # CutMix bounding box
    cut_w = int(W * math.sqrt(1 - lam))
    cut_h = int(H * math.sqrt(1 - lam))
    cx = random.randint(0, W)
    cy = random.randint(0, H)
    x1 = max(cx - cut_w // 2, 0)
    y1 = max(cy - cut_h // 2, 0)
    x2 = min(cx + cut_w // 2, W)
    y2 = min(cy + cut_h // 2, H)
    return x1, y1, x2, y2

def cutmix_batch(x, y, alpha=CUTMIX_ALPHA):
    if alpha <= 0: return x, y, y, 1.0
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    idx = torch.randperm(x.size(0), device=x.device)
    x1, y_a = x.clone(), y
    y_b = y[idx]
    _, _, H, W = x.size()
    xL, yT, xR, yB = rand_bbox(W, H, lam)
    x1[:, :, yT:yB, xL:xR] = x[idx, :, yT:yB, xL:xR]
    # adjust lam to the actually mixed area
    lam = 1 - ((xR - xL) * (yB - yT) / (W * H))
    return x1, y_a, y_b, lam

def criterion_mixed(logits, y_a, y_b, lam):
    return lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)

# ==========================
# Eval helpers
# ==========================
def evaluate(loader):
    model.eval()
    ce_sum, y_true, y_pred = 0.0, [], []
    with torch.no_grad():
        for x,y in loader:
            x,y = x.to(DEVICE), y.to(DEVICE)
            with amp_ctx:
                logits = model(x)
                loss = criterion(logits,y)
            ce_sum += loss.item()*x.size(0)
            y_true += y.cpu().tolist()
            y_pred += logits.argmax(1).cpu().tolist()
    f1 = f1_score(y_true,y_pred,average="macro")
    return ce_sum/len(loader.dataset), f1

def evaluate_test():
    return evaluate(te)

# ==========================
# Train
# ==========================
best = {"f1":0.0, "state":None, "epoch":0}
no_improve = 0
t0 = time.time()

for epoch in range(1, EPOCHS+1):
    epoch_start = time.time()

    # Unfreeze for fine-tuning after warmup (only deeper layers to keep it fast)
    if epoch == WARMUP_EPOCHS+1:
        # Unfreeze layer4 + FC (optionally layer3 too if you want a bit more adaptation)
        for p in model.layer4.parameters(): p.requires_grad = True
        # for p in model.layer3.parameters(): p.requires_grad = True  # uncomment if you want deeper unfreeze
        for p in model.fc.parameters(): p.requires_grad = True

        opt = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR_BODY, weight_decay=WD)
        # Cosine schedule after unfreezing
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=(EPOCHS - epoch + 1), eta_min=1e-6)
        print("🔓 Unfroze backbone (layer4) and switched to cosine LR.")

    # ---- train one epoch
    model.train()
    run_loss = 0.0
    for x,y in tr:
        x,y = x.to(DEVICE), y.to(DEVICE)
        opt.zero_grad(set_to_none=True)

        # Randomly apply MixUp OR CutMix
        do_aug = (random.random() < AUG_PROB)
        which = random.choice(["mixup","cutmix"]) if do_aug else "none"

        if which == "mixup" and USE_MIXUP:
            x_aug, y_a, y_b, lam = mixup_batch(x, y, MIXUP_ALPHA)
            with amp_ctx:
                logits = model(x_aug)
                loss = criterion_mixed(logits, y_a, y_b, lam)
        elif which == "cutmix" and USE_CUTMIX:
            x_aug, y_a, y_b, lam = cutmix_batch(x, y, CUTMIX_ALPHA)
            with amp_ctx:
                logits = model(x_aug)
                loss = criterion_mixed(logits, y_a, y_b, lam)
        else:
            with amp_ctx:
                logits = model(x)
                loss = criterion(logits,y)

        if scaler:  # CUDA path
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            opt.step()

        run_loss += loss.item()*x.size(0)

    # ---- validation
    v_loss, v_f1 = evaluate(vl)
    if sched is not None:
        sched.step()

    # timings
    epoch_time = time.time() - epoch_start
    total_time = time.time() - t0
    eta_min = (EPOCHS - epoch) * (epoch_time / 60)

    current_lr = opt.param_groups[0]["lr"]
    print(f"Epoch {epoch:02d} | train_loss={(run_loss/len(dtr)):.4f}  "
          f"val_loss={v_loss:.4f}  val_f1={v_f1:.4f}  "
          f"LR={current_lr:.6f}  time={epoch_time:.1f}s  total={total_time/60:.1f}m  ETA={eta_min:.1f}m")

    # early stopping tracking
    if v_f1 > best["f1"]:
        best = {"f1":v_f1, "state":copy.deepcopy(model.state_dict()), "epoch":epoch}
        no_improve = 0
    else:
        no_improve += 1
        if no_improve >= PATIENCE:
            print("⏹ Early stopping.")
            break

# ==========================
# Save & test
# ==========================
stamp = int(time.time())
ckpt_path = OUT / f"best_resnet50_{stamp}.pt"
torch.save({"state_dict":best["state"], "classes":classes, "arch":"resnet50"}, ckpt_path)
print(f"✅ Saved best from epoch {best['epoch']} (val_f1={best['f1']:.4f}) -> {ckpt_path}")

model.load_state_dict(best["state"])
t_loss, t_f1 = evaluate_test()
print(f"📊 TEST loss={t_loss:.4f}  TEST F1={t_f1:.4f}  (elapsed={(time.time()-t0)/60:.1f} min)")
