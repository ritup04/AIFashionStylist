# scripts/build_faiss.py
"""
Builds a FAISS similarity index using CLIP embeddings.
Automatically detects dataset and output paths.
"""

import os, pickle, clip, faiss, torch, numpy as np
from PIL import Image
from tqdm import tqdm

# ---------- PATHS ----------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG_DIR = os.path.join(BASE_DIR, "data_subset", "val")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

INDEX_PATH = os.path.join(MODEL_DIR, "fashion_index.faiss")
PATHS_PKL = os.path.join(MODEL_DIR, "image_paths.pkl")

# ---------- LOAD MODEL ----------
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# ---------- EMBEDDINGS ----------
image_files = []
for root, _, files in os.walk(IMG_DIR):
    for f in files:
        if f.lower().endswith(('.jpg', '.png', '.jpeg')):
            image_files.append(os.path.join(root, f))

embeddings = []
for path in tqdm(image_files, desc="Embedding images"):
    try:
        img = preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = model.encode_image(img)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        embeddings.append(emb.cpu().numpy())
    except Exception:
        pass  # skip unreadable images

embeddings = np.vstack(embeddings).astype('float32')
faiss.normalize_L2(embeddings)

# ---------- BUILD INDEX ----------
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)
faiss.write_index(index, INDEX_PATH)
pickle.dump(image_files, open(PATHS_PKL, "wb"))

print(f"✅ FAISS index built with {len(image_files)} images")
print(f"Saved index: {INDEX_PATH}")
