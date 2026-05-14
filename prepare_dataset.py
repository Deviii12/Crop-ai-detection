import shutil, sys
from pathlib import Path
import splitfolders

RAW_DIR        = Path("data/raw/PlantVillage")                     # original PlantVillage
AUG_BASE       = Path("data/New Plant Diseases Dataset(Augmented)")
AUG_TRAIN_DIR  = AUG_BASE / "train"
AUG_VAL_DIR    = AUG_BASE / "val"

MERGED_DIR = Path("data/merged_raw")
SPLIT_DIR  = Path("data/split")
VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def safe_rm(path: Path):
    if path.exists():
        shutil.rmtree(path)

def copy_tree(src_root: Path, dst_root: Path):
    if not src_root.exists():
        print(f"⚠️  Missing: {src_root}")
        return 0, 0
    copied = 0
    cls_dirs = [d for d in src_root.iterdir() if d.is_dir()]
    for cls in cls_dirs:
        dest = dst_root / cls.name
        dest.mkdir(parents=True, exist_ok=True)
        for f in cls.rglob("*"):
            if f.is_file() and f.suffix.lower() in VALID_EXTS:
                out = dest / f.name
                i = 1
                while out.exists():
                    out = dest / f"{out.stem}_{i}{out.suffix}"
                    i += 1
                shutil.copy2(f, out)
                copied += 1
    return copied, len(cls_dirs)

if __name__ == "__main__":
    # clean old
    safe_rm(MERGED_DIR); MERGED_DIR.mkdir(parents=True, exist_ok=True)
    safe_rm(SPLIT_DIR)

    tot = 0
    for src in [RAW_DIR, AUG_TRAIN_DIR, AUG_VAL_DIR]:
        c, k = copy_tree(src, MERGED_DIR)
        print(f"✅ Copied {c} images from {src} ({k} classes)")
        tot += c

    if tot == 0:
        print("❌ No images copied. Check paths at top.")
        sys.exit(1)

    print("🔪 Creating 80/10/10 split …")
    splitfolders.ratio(str(MERGED_DIR), output=str(SPLIT_DIR), seed=42, ratio=(0.8, 0.1, 0.1))
    print("✅ Done:", SPLIT_DIR)
