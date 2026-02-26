import os, sqlite3, random, datetime, re, time
from pathlib import Path
from flask import Flask, render_template, request, send_file
import torch
from torchvision import transforms, models
import torch.nn as nn
from PIL import Image
import wikipediaapi
from deep_translator import GoogleTranslator
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
import pandas as pd
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import numpy as np


HERE = Path(__file__).resolve()
SRC_DIR = HERE.parents[1]         
ROOT    = HERE.parents[2]   
DB_DIR  = SRC_DIR / "db"

MODELS_DIR = Path(os.environ.get("CROP_AI_MODELS", ROOT / "models"))

app = Flask(
    __name__,
    static_folder=str(HERE.parent / "static"),
    template_folder=str(HERE.parent / "templates")
)
UPLOAD_FOLDER = Path(app.static_folder) / "uploads"
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)

model_files = sorted(MODELS_DIR.glob("best_*.pt"))
if not model_files:
    raise FileNotFoundError(f"❌ No trained model found in {MODELS_DIR}")

CKPT = model_files[-1]
bundle = torch.load(CKPT, map_location="cpu")
CLASSES = bundle["classes"]
ARCH    = bundle.get("arch", "resnet50")
print(f"✅ Model loaded: {CKPT.name} ({ARCH}) with {len(CLASSES)} classes")

if ARCH == "efficientnet_b0":
    from torchvision.models import efficientnet_b0
    model = efficientnet_b0(weights=None)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, len(CLASSES))
elif ARCH == "resnet50":
    model = models.resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(CLASSES))
else:
    raise ValueError(f"❌ Unsupported arch: {ARCH}")

model.load_state_dict(bundle["state_dict"])
model.eval()


IMG_SIZE = 300 if ARCH == "efficientnet_b0" else 224
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

TTA_CANDIDATES = [
    transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(), transforms.Normalize(MEAN, STD)]),
    transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.RandomHorizontalFlip(p=1.0), transforms.ToTensor(), transforms.Normalize(MEAN, STD)]),
    transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.RandomRotation(15), transforms.ToTensor(), transforms.Normalize(MEAN, STD)]),
    transforms.Compose([transforms.Resize(int(IMG_SIZE*1.15)), transforms.CenterCrop(IMG_SIZE), transforms.ToTensor(), transforms.Normalize(MEAN, STD)]),
]

# Wikipedia setup
wiki = wikipediaapi.Wikipedia(language='en', user_agent="CropAI-StudentProject/1.0")

# -------------------------------------------------
# Utilities
# -------------------------------------------------
def clean_disease_name(name: str) -> str:
    return name.replace("___", " ").replace("__", " ").replace("_", " ").strip()

def _read_csv_safe(path: Path):
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"⚠️ CSV read error for {path}: {e}")
        return None

# # ---------- Crop name & fertilizer schedule ----------
# def base_crop_from_label(name: str) -> str:
#     """Return base crop from a PlantVillage-style label."""
#     s = clean_disease_name(name).strip()
#     if "(" in s and ")" in s:
#         # e.g., "corn (maize) cercospora leaf spot ..." -> "corn (maize)"
#         s = s[: s.find(")") + 1]
#     else:
#         # e.g., "grape black rot" -> "grape"
#         s = s.split()[0]
#     return s

# def get_fertilizer_schedule(crop_name: str):
#     """
#     Read src/db/fertilizer_schedules.csv and return list rows for given crop.
#     Each row: week, stage, npk, notes.
#     """
#     path = DB_DIR / "fertilizer_schedules.csv"
#     df = _read_csv_safe(path)
#     if df is None or "crop" not in df.columns:
#         return []
#     mask = df["crop"].astype(str).str.strip().str.lower() == crop_name.strip().lower()
#     rows = df[mask].copy()
#     if "week" in rows.columns:
#         rows["week"] = pd.to_numeric(rows["week"], errors="coerce")
#         rows = rows.sort_values("week", na_position="last")

#     out = []
#     for _, r in rows.iterrows():
#         n = r.get("npk_n", "")
#         p = r.get("npk_p2o5", "")
#         k = r.get("npk_k2o", "")
#         out.append({
#             "week": (int(r["week"]) if pd.notna(r.get("week")) else ""),
#             "stage": r.get("stage", ""),
#             "npk": f"{n}-{p}-{k}" if any([str(n), str(p), str(k)]) else "",
#             "notes": r.get("notes", "")
#         })
#     return out

# ---------- Matching helpers ----------
def _norm_key(s: str) -> str:
    if not s:
        return ""
    s = clean_disease_name(str(s)).lower()
    s = s.replace("grey", "gray")
    s = re.sub(r"[()\-.,/]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _token_set_score(a: str, b: str) -> float:
    A = set(_norm_key(a).split())
    B = set(_norm_key(b).split())
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)

ALIAS_MAP = {
    _norm_key("corn (maize) cercospora leaf spot gray leaf spot"): _norm_key("corn (maize) cercospora leaf spot"),
    _norm_key("corn (maize) northern leaf blight"): _norm_key("corn (maize) northern leaf blight"),
    _norm_key("corn (maize) common rust"): _norm_key("corn (maize) common rust"),
}

def _build_map(df: pd.DataFrame):
    if df is None or "disease_name" not in df.columns:
        return {}
    return {_norm_key(r["disease_name"]): r for _, r in df.iterrows()}

def _best_fuzzy_key(query_key: str, key_map: dict, thresh=0.60):
    best_k, best_s = None, 0.0
    for k2 in key_map.keys():
        s = _token_set_score(query_key, k2)
        if s > best_s:
            best_k, best_s = k2, s
    return best_k if best_s >= thresh else None

# ---------- Image checks ----------
def is_leaf(image_path: str) -> bool:
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    green_pixels = sum(1 for px in img.getdata() if px[1] > px[0] and px[1] > px[2])
    return green_pixels > (0.2 * w * h)

# ---------- Prediction + TTA ----------
def predict_image_tta(image_path, threshold=70.0, topk=3, n_aug=8, temp=0.7):
    """
    Returns:
      display_label (str),
      predictions (list[(class_name, prob%)]),
      top_idx (int)  <-- index in CLASSES of the top-1 class
    """
    img = Image.open(image_path).convert("RGB")
    with torch.no_grad():
        probs_sum = torch.zeros(len(CLASSES))
        for _ in range(n_aug):
            T_aug = random.choice(TTA_CANDIDATES)
            x = T_aug(img).unsqueeze(0)
            logits = model(x)
            probs = torch.softmax(logits / temp, dim=1).squeeze(0)
            probs_sum += probs
        avg_probs = probs_sum / n_aug
        values, indices = torch.topk(avg_probs, topk)
        predictions = [(CLASSES[i.item()], float(values[j].item()) * 100.0) for j, i in enumerate(indices)]
        top_idx = indices[0].item()
        top_label = CLASSES[top_idx]
        top_conf = predictions[0][1]
        display_label = top_label if top_conf >= threshold else f"Uncertain (best guess: {top_label})"
        return display_label, predictions, top_idx

# ---------- Grad-CAM ----------
def _find_last_conv_layer(m: nn.Module) -> nn.Module | None:
    """Find the last nn.Conv2d in the model."""
    last = None
    for _name, module in m.named_modules():
        if isinstance(module, nn.Conv2d):
            last = module
    return last

def generate_gradcam(image_path: str, model, target_class_idx: int) -> str | None:
    """
    Generate Grad-CAM heatmap for the target class.
    Uses context manager API (no use_cuda kw). Temporarily enables grads.
    """
    try:
        img = Image.open(image_path).convert("RGB")
        preprocess = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ])
        input_tensor = preprocess(img).unsqueeze(0)

        target_layer = _find_last_conv_layer(model)
        if target_layer is None:
            print("⚠️ Grad-CAM: could not locate a Conv2d layer.")
            return None

        with torch.enable_grad():
            model.zero_grad(set_to_none=True)
            with GradCAM(model=model, target_layers=[target_layer]) as cam:
                grayscale_cam = cam(
                    input_tensor=input_tensor,
                    targets=[ClassifierOutputTarget(int(target_class_idx))]
                )[0, :]

        img_np = np.array(img.resize((IMG_SIZE, IMG_SIZE)), dtype=np.float32) / 255.0
        cam_image = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)

        out_dir = Path(app.static_folder) / "uploads"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_name = f"gradcam_{int(time.time()*1000)}.jpg"
        out_path = out_dir / out_name
        Image.fromarray(cam_image).save(out_path)
        print("✅ Grad-CAM saved:", out_path)
        return out_name
    except Exception as e:
        print("⚠️ Grad-CAM generation failed:", e)
        return None

# ---------- Treatment lookup ----------
def get_treatment(raw_label: str):
    """
    Lookup treatment from treatments.csv + mediciness.csv by normalized disease name.
    - exact normalized match
    - alias map to canonical
    - fuzzy token-set match (>=0.60)
    - wikipedia fallback
    """
    treatments_csv = DB_DIR / "treatments.csv"
    medicines_csv  = DB_DIR / "mediciness.csv"   # <-- KEEP THIS NAME

    raw_label_disp = clean_disease_name(raw_label)
    key = _norm_key(raw_label_disp)

    tdf = _read_csv_safe(treatments_csv)
    mdf = _read_csv_safe(medicines_csv)
    tmap = _build_map(tdf)
    mmap = _build_map(mdf)

    match_key = key if key in tmap else None
    if match_key is None and key in ALIAS_MAP:
        alias_key = ALIAS_MAP[key]
        if alias_key in tmap:
            match_key = alias_key
    if match_key is None and tmap:
        maybe = _best_fuzzy_key(key, tmap, thresh=0.60)
        if maybe:
            match_key = maybe

    if match_key is not None:
        r = tmap[match_key]
        treatment_info = {
            "organic": r.get("organic", "No info"),
            "chemical": r.get("chemical", "No info"),
            "prevention": r.get("prevention", "No info"),
            "symptoms": r.get("symptoms", "No info"),
            "climate_risk": r.get("climate_risk", "No info"),
            "fertilizer_guidelines": r.get("fertilizer_guidelines", "No info"),
            "notes": r.get("notes", "No info"),
            "source": "csv"
        }

        organic_meds, chemical_meds = "No info", "No info"
        if mmap:
            mr = mmap.get(match_key)
            if mr is None:
                mk = _best_fuzzy_key(match_key, mmap, thresh=0.60)
                if mk:
                    mr = mmap[mk]
            if mr is not None:
                organic_meds  = mr.get("organic_medicines", "No info")
                chemical_meds = mr.get("chemical_medicines", "No info")

        treatment_info["organic_medicines"]  = organic_meds
        treatment_info["chemical_medicines"] = chemical_meds
        return treatment_info

    # Wikipedia fallback
    page = wiki.page(raw_label_disp)
    if page.exists():
        summary = page.summary[:500] + "..." if len(page.summary) > 500 else page.summary
        return {
            "organic": summary, "organic_medicines": "No info",
            "chemical": summary, "chemical_medicines": "No info",
            "prevention": summary, "symptoms": "No info",
            "climate_risk": "No info", "fertilizer_guidelines": "No info",
            "notes": "Auto-fetched from Wikipedia", "source": "wikipedia"
        }

    return {
        "organic":"No info","organic_medicines":"No info",
        "chemical":"No info","chemical_medicines":"No info",
        "prevention":"No info","symptoms":"No info",
        "climate_risk":"No info","fertilizer_guidelines":"No info",
        "notes":"No info","source":"none"
    }

# ---------- Translation ----------
def translate_text(text, lang="en"):
    if lang == "en" or not text:
        return text
    try:
        return GoogleTranslator(source="en", target=lang).translate(text)
    except Exception:
        return text

def translate_treatment(treatment: dict, lang="en"):
    if lang == "en":
        return treatment
    return {k: (translate_text(v, lang) if isinstance(v, str) else v) for k, v in treatment.items()}

# ---------- Report ----------
def save_report(disease, predictions, treatment, filename):
    doc = SimpleDocTemplate(filename, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []
    elements.append(Paragraph(f"<b>Disease:</b> {disease}", styles['Normal']))
    elements.append(Spacer(1, 12))

    pred_data = [["Label", "Confidence (%)"]]
    for name, conf in predictions:
        pred_data.append([name, f"{conf:.2f}"])
    table = Table(pred_data, hAlign="LEFT")
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.lightgreen),
        ('GRID', (0,0), (-1,-1), 0.5, colors.black),
    ]))
    elements.append(table)
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("<b>Organic Treatment:</b>", styles['Normal']))
    elements.append(Paragraph(treatment.get('organic', "No info"), styles['Normal']))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("<b>Chemical Treatment:</b>", styles['Normal']))
    elements.append(Paragraph(treatment.get('chemical', "No info"), styles['Normal']))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("<b>Prevention:</b>", styles['Normal']))
    elements.append(Paragraph(treatment.get('prevention', "No info"), styles['Normal']))

    doc.build(elements)
    return filename

# -------------------------------------------------
# DB bootstrap (community)
# -------------------------------------------------
def ensure_db():
    db_path = DB_DIR / "cropcare.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(db_path))
    cur = con.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS community_posts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            disease_name TEXT,
            notes TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    con.commit()
    con.close()

# ===============================
# Routes
# ===============================
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("leaf")
        lang = request.form.get("lang", "en")
        city = request.form.get("city", "Delhi").strip() or "Delhi"

        if file and file.filename:
            filepath = Path(app.config["UPLOAD_FOLDER"]) / file.filename
            file.save(str(filepath))

            if not is_leaf(str(filepath)):
                return "❌ Please upload a clear leaf image."

            # Predict (also returns top_idx)
            display_label, predictions, top_idx = predict_image_tta(str(filepath))

            # Use true top-1 class for treatment & Grad-CAM
            top1_label_raw = CLASSES[top_idx]
            top1_label     = clean_disease_name(top1_label_raw)

            # Lookup and translate treatment
            treatment_raw = get_treatment(top1_label)
            treatment = translate_treatment(treatment_raw, lang)

            # Grad-CAM heatmap for the top class
            gradcam_filename = generate_gradcam(str(filepath), model, top_idx)

            # Fertilizer schedule for the base crop
            # crop_name = base_crop_from_label(top1_label)
            # fert_schedule = get_fertilizer_schedule(crop_name)

            # Translate labels for UI
            display_label_clean = clean_disease_name(display_label)
            label_translated = translate_text(display_label_clean, lang)
            predictions_translated = [
                (translate_text(clean_disease_name(name), lang), conf)
                for name, conf in predictions
            ]

            # ---- translated helper text for dosage + organic/chemical block ----
            dose_heading = translate_text("🔹 a. Dosage Calculator", lang)
            dose_help = translate_text(
                "How to use:\n"
                "1) Enter your field size and select acre / hectare.\n"
                "2) Type the recommended dose from the label (for example \"2 ml per litre\").\n"
                "3) Add your spray volume (how many litres of water you normally spray per acre or hectare).\n"
                "4) The tool will show the total product and total water needed for your field.",
                lang
            )
            orgchem_heading = translate_text("🔹 b. Organic vs Chemical Comparison", lang)
            orgchem_intro = translate_text(
                "Quick view to decide which option fits today. Organic is usually safer and good for prevention. "
                "Chemical is usually stronger and fast for serious attack, but needs more care.",
                lang
            )

            # Report
            report_path = Path(app.static_folder) / "report.pdf"
            save_report(label_translated, predictions_translated, treatment, str(report_path))

            return render_template(
                "result.html",
                filename=file.filename,
                disease=label_translated,
                predictions=predictions_translated,
                treatment=treatment,
                report_url="/download_report",
                city=city,
                gradcam_filename=gradcam_filename,  # show CAM if present
                # crop_name=crop_name,
                # fert_schedule=fert_schedule,
                dose_heading=dose_heading,
                dose_help=dose_help,
                orgchem_heading=orgchem_heading,
                orgchem_intro=orgchem_intro,
                lang=lang,
            )
    return render_template("index.html")

@app.route("/download_report")
def download_report():
    report_path = Path(app.static_folder) / "report.pdf"
    if not report_path.exists():
        return "⚠️ Report not found. Please generate one first."
    as_of = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return send_file(str(report_path), as_attachment=True, download_name=f"CropAI_Report_{as_of}.pdf")

@app.route("/community", methods=["GET", "POST"])
def community():
    ensure_db()
    db_path = DB_DIR / "cropcare.db"
    con = sqlite3.connect(str(db_path))
    cur = con.cursor()
    if request.method == "POST":
        disease = request.form.get("disease", "").strip()
        notes = request.form.get("notes", "").strip()
        if disease or notes:
            cur.execute("INSERT INTO community_posts (disease_name, notes) VALUES (?,?)", (disease, notes))
            con.commit()
    cur.execute("SELECT disease_name, notes, created_at FROM community_posts ORDER BY created_at DESC")
    posts = cur.fetchall()
    con.close()
    return render_template("community.html", posts=posts)

if __name__ == "__main__":
    ensure_db()
    UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
    app.run(host="0.0.0.0", port=5000, debug=True)
