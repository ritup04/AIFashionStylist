# demo_app.py
"""
AI Fashion Stylist — Occasion-aware, color- and category-smart recommender
"""

import os
import torch
import clip
import faiss
import pickle
import pandas as pd
from torchvision import transforms, models
from PIL import Image, ImageDraw
import streamlit as st
from scripts.color_utils import get_dominant_colors
import webcolors
import numpy as np
from colorsys import rgb_to_hsv, hsv_to_rgb

# ---------------------------
# Utilities
# ---------------------------

def closest_color_name(rgb_tuple):
    """Return closest CSS3 color name, robust to webcolors versions."""
    try:
        return webcolors.rgb_to_name(tuple(map(int, rgb_tuple)))
    except Exception:
        # fallback: build CSS3 map
        try:
            css_map = webcolors.CSS3_NAMES_TO_HEX
        except Exception:
            css_map = {name: webcolors.name_to_hex(name) for name in webcolors.names("css3")}
        best = None
        best_dist = float("inf")
        for name, hexv in css_map.items():
            try:
                rc, gc, bc = webcolors.hex_to_rgb(hexv)
            except Exception:
                continue
            dist = (rc - rgb_tuple[0])**2 + (gc - rgb_tuple[1])**2 + (bc - rgb_tuple[2])**2
            if dist < best_dist:
                best_dist = dist
                best = name
        return best or "unknown"

def map_category_to_group(cat_name):
    """Map detailed category name -> high level group."""
    if not isinstance(cat_name, str):
        return "unknown"
    c = cat_name.lower().strip()
    if any(x in c for x in ["top", "shirt", "blouse", "t-shirt", "hoodie", "kurti", "sweater", "tank"]):
        return "topwear"
    if any(x in c for x in ["jean", "pant", "trouser", "skirt", "short", "bottom", "leggings"]):
        return "bottomwear"
    if any(x in c for x in ["dress", "jumpsuit", "frock", "gown", "outfit"]):
        return "dress"
    if any(x in c for x in ["coat", "jacket", "blazer", "outerwear"]):
        return "outerwear"
    return "unknown"

def generate_color_harmony(rgb):
    """Return list: [complementary, analogous1, analogous2] in RGB int tuples."""
    r, g, b = [x/255.0 for x in rgb]
    h, s, v = rgb_to_hsv(r, g, b)
    # complementary
    comp_h = (h + 0.5) % 1.0
    analogs = [(h + 0.08) % 1.0, (h - 0.08) % 1.0]
    out = []
    def hsv_to_rgb255(hh):
        rr, gg, bb = hsv_to_rgb(hh, s, v)
        return (int(rr*255), int(gg*255), int(bb*255))
    out.append(hsv_to_rgb255(comp_h))
    out.extend(hsv_to_rgb255(a) for a in analogs)
    return out

def get_color_tone(rgb):
    """Return 'dark'|'light'|'vibrant'|'neutral' based on simple heuristics."""
    r, g, b = rgb
    brightness = (r + g + b) / 3
    saturation = max(r, g, b) - min(r, g, b)
    if brightness < 80:
        return "dark"
    if brightness > 200:
        return "light"
    if saturation > 100:
        return "vibrant"
    return "neutral"

# ---------------------------
# Styling / Occasion logic
# ---------------------------

def suggest_matching_items(outfit_type, colors, occasion):
    """Return a list of formatted strings (markdown-ready) with styling advice."""
    # colors: list of RGB tuples
    readable = [closest_color_name(c).title() for c in colors]
    tone_list = [get_color_tone(c) for c in colors]
    dominant_tone = max(set(tone_list), key=tone_list.count)
    color_text = ", ".join(readable)
    suggestions = []
    suggestions.append(f"🎨 **Detected palette:** {color_text} — *{dominant_tone.title()} tone*")
    # Occasion-aware base text
    if outfit_type == "topwear":
        suggestions.append(f"👚 **Topwear styling ({occasion})** — Try:")
        if occasion == "Casual":
            suggestions += [
                "- High-waisted jeans or a pleated skirt.",
                "- White sneakers or simple flats.",
                "- Minimal jewelry (studs, thin chain)."
            ]
        elif occasion == "Formal":
            suggestions += [
                "- Tailored trousers or a pencil skirt.",
                "- Blazer in neutral shade, closed shoes.",
                "- Delicate gold/silver jewelry."
            ]
        elif occasion == "Party":
            suggestions += [
                "- Leather pants or sequin skirt for glam.",
                "- Statement earrings and heels.",
                "- Consider metallic clutch or bold lip."
            ]
        elif occasion == "Date":
            suggestions += [
                "- Soft skirts or fitted jeans.",
                "- Romantic layers (shrug/cardigan).",
                "- Subtle sparkle (small pendant)."
            ]
        elif occasion == "Streetwear":
            suggestions += [
                "- Oversized jacket or cargo pants.",
                "- Chunky trainers and layered chains.",
                "- Cap or beanie for attitude."
            ]
        elif occasion == "Festive":
            suggestions += [
                "- Embellished skirt or ethnic bottoms.",
                "- Bold earrings (jhumka / chandbali).",
                "- Bright handbag or dupatta to pop."
            ]
    elif outfit_type == "bottomwear":
        suggestions.append(f"👖 **Bottomwear styling ({occasion})** — Try:")
        if occasion == "Casual":
            suggestions += [
                "- Neutral or graphic tees tucked in.",
                "- Denim jacket / bomber for layers.",
                "- Clean sneakers or sandals."
            ]
        elif occasion == "Formal":
            suggestions += [
                "- Crisp shirt or blouse tucked in.",
                "- Blazer and polished shoes.",
                "- Minimal accessories."
            ]
        elif occasion == "Party":
            suggestions += [
                "- Crop tops or shimmering blouses.",
                "- Heels & clutch to elevate the look.",
                "- Bold makeup if desired."
            ]
        elif occasion == "Date":
            suggestions += [
                "- Romantic tops, soft fabrics.",
                "- Heels or nice flats.",
                "- Delicate jewelry."
            ]
        elif occasion == "Streetwear":
            suggestions += [
                "- Oversized tees, hoodies, or layered tops.",
                "- Statement sneakers and cap.",
            ]
        elif occasion == "Festive":
            suggestions += [
                "- Embroidered / festive tops.",
                "- Traditional footwear or embellished flats.",
            ]
    elif outfit_type == "dress":
        suggestions.append(f"👗 **Dress styling ({occasion})** — Try:")
        if occasion == "Casual":
            suggestions += [
                "- Pair with sneakers or sandals.",
                "- Light jacket for layering."
            ]
        elif occasion == "Formal":
            suggestions += [
                "- Pumps and a structured bag.",
                "- Fine jewelry and neutral dupatta/scarf."
            ]
        elif occasion == "Party":
            suggestions += [
                "- Statement jewelry and heels.",
                "- Bold clutch and silhouette-defining belt."
            ]
        elif occasion == "Date":
            suggestions += [
                "- Romantic soft makeup, delicate accessories.",
                "- Neutral heels or wedges."
            ]
        elif occasion == "Streetwear":
            suggestions += [
                "- Layer with oversized denim or leather jacket.",
                "- Boots or trainers for edge."
            ]
        elif occasion == "Festive":
            suggestions += [
                "- Embellished overlays or colorful shawls.",
                "- Traditional jewelry to enhance festive vibe."
            ]
    else:
        suggestions.append(f"🧥 **Outerwear / Unknown styling ({occasion})** — Try general tips:")
        suggestions += [
            "- Keep inner pieces simple and balanced.",
            "- Choose footwear that matches the overall tone.",
            "- Accessorize depending on occasion (minimal → formal, bold → party)."
        ]
    # Color harmony advice (generic)
    suggestions.append("🎯 **Color harmony tips:**")
    if dominant_tone == "dark":
        suggestions.append("- Try lighter or pastel bottoms/tops to balance the darkness.")
    elif dominant_tone == "light":
        suggestions.append("- Dark bottoms/tops provide a stylish contrast.")
    elif dominant_tone == "vibrant":
        suggestions.append("- Use neutral pieces to ground vibrant tones, or pair with complementary colors for a bold look.")
    else:
        suggestions.append("- Neutral palettes work well with textured or patterned pieces to add interest.")
    return suggestions

# ---------------------------
# Streamlit UI
# ---------------------------

st.set_page_config(page_title="AI Fashion Stylist", page_icon="👗", layout="centered")
st.title("👗 AI Fashion Stylist")

uploaded = st.file_uploader("Upload your outfit image", type=["jpg", "jpeg", "png"])
if not uploaded:
    st.info("Upload an outfit image to get personalized styling and similar items.")
    st.stop()

# Occasion selector (user chooses)
occasion = st.selectbox("Select occasion or mood:", ["Casual", "Formal", "Party", "Date", "Streetwear", "Festive"])

img = Image.open(uploaded).convert("RGB")
st.image(img, caption="Uploaded Image", use_container_width=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Paths / models
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "DeepFashion2", "img_info_dataframes")

# Load CSVs (train/val/test if present)
csv_paths = [os.path.join(DATA_DIR, f) for f in ["train.csv", "validation.csv", "test.csv"] if os.path.exists(os.path.join(DATA_DIR, f))]
if len(csv_paths) == 0:
    st.warning("No csv dataframes found in DeepFashion2/img_info_dataframes — complementary filtering will be limited.")
    df = pd.DataFrame(columns=["path", "category_name"])
else:
    df = pd.concat([pd.read_csv(p) for p in csv_paths], ignore_index=True)
    df["category_name"] = df.get("category_name", "").fillna("unknown")
    df["category_group"] = df["category_name"].apply(map_category_to_group)

# Load classifier checkpoint
clf_files = [f for f in os.listdir(MODEL_DIR) if f.startswith("cloth_classifier_")]
clf_files.sort(reverse=True)
if len(clf_files) == 0:
    st.error("No cloth classifier found in models/. Please place your classifier checkpoint named cloth_classifier_*.")
    st.stop()
CLASSIFIER_PATH = os.path.join(MODEL_DIR, clf_files[0])

ckpt = torch.load(CLASSIFIER_PATH, map_location=device)
model = models.resnet50(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, len(ckpt["classes"]))
model.load_state_dict(ckpt["model"])
model = model.to(device).eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Predict label & map to group
with torch.no_grad():
    pred_idx = model(transform(img).unsqueeze(0).to(device)).argmax(1).item()
pred_label = ckpt["classes"][pred_idx]
outfit_type = map_category_to_group(pred_label)
st.subheader(f"👕 Detected: {pred_label.title()} ({outfit_type.title()})")

# Dominant colors
st.subheader("🎨 Dominant Colors")
temp_path = os.path.join(MODEL_DIR, "temp_query.jpg")
img.save(temp_path)
colors = get_dominant_colors(temp_path, k=3)  # list of (r,g,b) tuples
readable = [closest_color_name(c).title() for c in colors]

cols = st.columns(len(colors))
for i, c in enumerate(colors):
    sw = Image.new("RGB", (120, 80), tuple(map(int, c)))
    draw = ImageDraw.Draw(sw)
    draw.rectangle([(0,0), (119,79)], outline=(200,200,200), width=2)
    cols[i].image(sw, caption=readable[i], use_container_width=True)

# FAISS retrieval
INDEX_PATH = os.path.join(MODEL_DIR, "fashion_index.faiss")
PATHS_PKL = os.path.join(MODEL_DIR, "image_paths.pkl")
similar_paths = []

if os.path.exists(INDEX_PATH) and os.path.exists(PATHS_PKL):
    index = faiss.read_index(INDEX_PATH)
    paths = pickle.load(open(PATHS_PKL, "rb"))
    clip_model, preprocess = clip.load("ViT-B/32", device=device)

    with torch.no_grad():
        q = preprocess(img).unsqueeze(0).to(device)
        qemb = clip_model.encode_image(q)
        qemb = qemb / qemb.norm(dim=-1, keepdim=True)
    qn = qemb.cpu().numpy().astype("float32")
    faiss.normalize_L2(qn)
    D, I = index.search(qn, k=200)  # retrieve many; we'll filter

    # complementary target group
    if outfit_type == "topwear":
        target_group = "bottomwear"
    elif outfit_type == "bottomwear":
        target_group = "topwear"
    elif outfit_type == "dress":
        target_group = "dress"
    else:
        target_group = None  # outerwear/unknown -> just show similar

    # Attempt to filter by category_group from CSV using basename matching
    shown = 0
    st.subheader("🛍️ Top 5 Similar Items")
    cols_disp = st.columns(5)

    # compute harmony names (optional advanced filter)
    harmony_rgbs = []
    for c in colors:
        harmony_rgbs.extend(generate_color_harmony(c))
    harmony_names = {closest_color_name(c).lower() for c in harmony_rgbs}

    # First pass: category match + (optional) color-harmony hint in path/category_name
    for idx in I[0]:
        if shown >= 5:
            break
        p = paths[idx]
        base = os.path.basename(p)
        matched_rows = df[df["path"].str.contains(base, case=False, na=False)]
        row_group = None
        row_cat_name = ""
        if len(matched_rows) > 0:
            row_group = matched_rows.iloc[0].get("category_group", None)
            row_cat_name = matched_rows.iloc[0].get("category_name", "")
        # If target_group set, require match; else accept any
        accept = False
        if target_group is None:
            accept = True
        else:
            if row_group == target_group:
                accept = True
        # Further prefer items that match color harmony names (best-effort)
        if accept:
            # if harmony name appears in category_name or path -> prefer it
            low_cat = str(row_cat_name).lower()
            if any(h in low_cat for h in harmony_names) or any(h in base.lower() for h in harmony_names):
                cols_disp[shown].image(p, use_container_width=True)
                shown += 1
            else:
                # keep as candidate but show later if needed
                similar_paths.append(p)
    # Fill remaining slots from similar_paths or straight from I
    i_ptr = 0
    while shown < 5:
        if similar_paths and i_ptr < len(similar_paths):
            cols_disp[shown].image(similar_paths[i_ptr], use_container_width=True)
            shown += 1
            i_ptr += 1
            continue
        # fallback: show the top visually-similar items irrespective of category
        for idx in I[0]:
            if shown >= 5:
                break
            p = paths[idx]
            # avoid duplicates
            if p in similar_paths:
                continue
            cols_disp[shown].image(p, use_container_width=True)
            shown += 1
        break

    if shown == 0:
        st.info("No similar items found (empty index or dataset).")
else:
    st.warning("⚠️ FAISS index or image_paths.pkl not found. Showing no similar items.")

# Dynamic styling suggestions (occasion-aware + color-aware)
st.subheader("💡 Styling Suggestions")
styling_suggestions = suggest_matching_items(outfit_type, colors, occasion)
for s in styling_suggestions:
    st.markdown(f"- {s}")

# cleanup
try:
    os.remove(temp_path)
except Exception:
    pass
