# evaluate_model.py
"""
Evaluate CROP-AI model on a test set and generate:
- evaluation_report.csv   (precision, recall, f1-score, support)
- confusion_matrix.png
- f1_scores.png

Run from project root:
    python evaluate_model.py
"""

import itertools
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms, datasets

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_recall_fscore_support,
)

import matplotlib.pyplot as plt
import pandas as pd


# ---------------- Paths ----------------
HERE = Path(__file__).resolve()
ROOT = HERE.parent                         # crop-ai/
MODELS_DIR = ROOT / "models"
TEST_DIR = ROOT / "data" / "test"          # <-- change if your test folder is elsewhere
OUT_DIR = ROOT / "reports"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------- Load model ----------------
model_files = sorted(MODELS_DIR.glob("best_*.pt"))
if not model_files:
    raise FileNotFoundError(f"No best_*.pt found in {MODELS_DIR}")

ckpt_path = model_files[-1]
bundle = torch.load(ckpt_path, map_location="cpu")

CLASSES = bundle["classes"]
ARCH = bundle.get("arch", "resnet50")
print(f"Loaded checkpoint: {ckpt_path.name}  (arch={ARCH}, classes={len(CLASSES)})")

if ARCH == "efficientnet_b0":
    from torchvision.models import efficientnet_b0
    model = efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(CLASSES))
elif ARCH == "resnet50":
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
else:
    raise ValueError(f"Unsupported arch: {ARCH}")

model.load_state_dict(bundle["state_dict"])
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Using device: {device}")

# ---------------- Data ----------------
IMG_SIZE = 300 if ARCH == "efficientnet_b0" else 224
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

test_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

if not TEST_DIR.exists():
    raise FileNotFoundError(f"Test directory not found: {TEST_DIR}")

test_ds = datasets.ImageFolder(TEST_DIR, transform=test_tfms)
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=32,
                                          shuffle=False, num_workers=2)

print(f"Test samples: {len(test_ds)}")
print("Classes from dataset:", test_ds.classes)
print("Classes from checkpoint:", CLASSES)

# If ordering is different, build a mapping
idx_map = []
for ds_class in test_ds.classes:
    if ds_class not in CLASSES:
        raise ValueError(f"Class '{ds_class}' not found in checkpoint CLASSES.")
    idx_map.append(CLASSES.index(ds_class))

criterion = nn.CrossEntropyLoss(reduction="sum")

all_labels = []
all_preds = []
all_probs = []
running_loss = 0.0

with torch.no_grad():
    for x, y in test_loader:
        x = x.to(device)
        # map dataset labels to checkpoint index space
        y_mapped = torch.tensor([idx_map[t.item()] for t in y],
                                dtype=torch.long, device=device)

        logits = model(x)
        loss = criterion(logits, y_mapped)
        running_loss += loss.item()

        probs = torch.softmax(logits, dim=1)
        pred = probs.argmax(dim=1)

        all_labels.extend(y_mapped.cpu().numpy())
        all_preds.extend(pred.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

n_samples = len(test_ds)
avg_loss = running_loss / n_samples
acc = accuracy_score(all_labels, all_preds)

print(f"\nTest loss: {avg_loss:.4f}")
print(f"Test accuracy: {acc*100:.2f}%")

# ---------------- Metrics & CSV ----------------
precision, recall, f1, support = precision_recall_fscore_support(
    all_labels, all_preds, labels=list(range(len(CLASSES)))
)

report_df = pd.DataFrame({
    "class": CLASSES,
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
    "support": support,
})

overall_row = pd.DataFrame({
    "class": ["OVERALL"],
    "precision": [precision.mean()],
    "recall": [recall.mean()],
    "f1_score": [f1.mean()],
    "support": [support.sum()],
})

report_df = pd.concat([report_df, overall_row], ignore_index=True)
csv_path = OUT_DIR / "evaluation_report.csv"
report_df.to_csv(csv_path, index=False)
print(f"Saved metrics to {csv_path}")

print("\nClassification report:\n")
print(classification_report(all_labels, all_preds, target_names=CLASSES))

# ---------------- Confusion matrix ----------------
cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(CLASSES))))
cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(cm_norm, interpolation="nearest", cmap=plt.cm.Blues)
ax.figure.colorbar(im, ax=ax)
ax.set(
    xticks=np.arange(len(CLASSES)),
    yticks=np.arange(len(CLASSES)),
    xticklabels=CLASSES,
    yticklabels=CLASSES,
    ylabel="True label",
    xlabel="Predicted label",
    title="Normalized Confusion Matrix",
)
plt.setp(ax.get_xticklabels(), rotation=90, ha="center",
         rotation_mode="anchor", fontsize=8)
plt.setp(ax.get_yticklabels(), fontsize=8)

# small text inside each cell
fmt = ".2f"
thresh = cm_norm.max() / 2.0
for i, j in itertools.product(range(cm_norm.shape[0]), range(cm_norm.shape[1])):
    ax.text(j, i, format(cm_norm[i, j], fmt),
            ha="center", va="center",
            color="white" if cm_norm[i, j] > thresh else "black",
            fontsize=6)

fig.tight_layout()
cm_path = OUT_DIR / "confusion_matrix.png"
plt.savefig(cm_path, dpi=300)
plt.close(fig)
print(f"Saved confusion matrix to {cm_path}")

# ---------------- F1-score bar chart ----------------
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(range(len(CLASSES)), f1)
ax.set_xticks(range(len(CLASSES)))
ax.set_xticklabels(CLASSES, rotation=90, fontsize=8)
ax.set_ylabel("F1-score")
ax.set_title("Per-class F1-score")
ax.set_ylim(0.0, 1.0)
f1_path = OUT_DIR / "f1_scores.png"
plt.tight_layout()
plt.savefig(f1_path, dpi=300)
plt.close(fig)
print(f"Saved F1-score plot to {f1_path}")

print("\nDone.")
