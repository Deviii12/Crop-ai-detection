import shutil, random, os
from pathlib import Path
from sklearn.model_selection import train_test_split

RAW = Path("data/raw")
OUT = Path("data/split")
random.seed(42)

def gather():
    items = []
    for cls in sorted(p.name for p in RAW.iterdir() if p.is_dir()):
        for img in (RAW/cls).glob("*.*"):
            items.append((str(img), cls))
    return items

def copy_list(pairs, subset):
    for src, cls in pairs:
        dst_dir = OUT/subset/cls
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst_dir/dst_name(Path(src).name, dst_dir))

def dst_name(name, dst_dir):
    n = name
    i = 1
    while (dst_dir/n).exists():
        stem, ext = os.path.splitext(name)
        n = f"{stem}_{i}{ext}"
        i += 1
    return n

if __name__ == "__main__":
    items = gather()
    X = [p for p,_ in items]
    y = [c for _,c in items]
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)
    X_v,  X_te, y_v, y_te = train_test_split(X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=42)
    copy_list(list(zip(X_tr,y_tr)), "train")
    copy_list(list(zip(X_v, y_v)),  "val")
    copy_list(list(zip(X_te,y_te)),  "test")
    print("Dataset split complete.")
