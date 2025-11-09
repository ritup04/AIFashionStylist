import os
import pandas as pd
import shutil
from tqdm import tqdm

# -------------------------
# Paths (relative)
# -------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(BASE_DIR, "DeepFashion2", "deepfashion2_original_images")
INFO_DIR = os.path.join(BASE_DIR, "DeepFashion2", "img_info_dataframes")

TRAIN_CSV = os.path.join(INFO_DIR, "train.csv")
VAL_CSV = os.path.join(INFO_DIR, "validation.csv")

TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "train", "image")
VAL_IMG_DIR = os.path.join(DATA_ROOT, "validation", "image")
OUTPUT_DIR = os.path.join(BASE_DIR, "data_subset")

TRAIN_SAMPLE, VAL_SAMPLE = 2000, 500

# -------------------------
# Category Simplification
# -------------------------
CATEGORY_MAP = {
    'shirt': 'top',
    't-shirt': 'top',
    'blouse': 'top',
    'sweater': 'top',
    'hoodie': 'top',
    'jacket': 'outerwear',
    'coat': 'outerwear',
    'down coat': 'outerwear',
    'vest': 'outerwear',
    'pants': 'bottom',
    'jeans': 'bottom',
    'shorts': 'bottom',
    'skirt': 'bottom',
    'dress': 'dress',
    'romper': 'dress',
    'long sleeve top': 'top',
    'short sleeve top': 'top',
    'long sleeve outwear': 'outerwear',
    'short sleeve dress': 'dress',
    'vest dress': 'dress',
    'shorts': 'bottom',
    'trousers': 'bottom'
}

# -------------------------
# Helper Functions
# -------------------------
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def map_category(name):
    if isinstance(name, str):
        for k, v in CATEGORY_MAP.items():
            if k in name.lower():
                return v
    return None


def copy_split(df, src_dir, dst_dir, sample_size, split_name):
    ensure_dir(dst_dir)
    df = df.dropna(subset=["path", "category_name"])

    df["broad_class"] = df["category_name"].apply(map_category)
    df = df.dropna(subset=["broad_class"])

    if sample_size < len(df):
        df = df.sample(sample_size, random_state=42)

    print(f"\n📂 Copying {len(df)} images for {split_name} ...")

    for cls in df["broad_class"].unique():
        ensure_dir(os.path.join(dst_dir, cls))

    for _, row in tqdm(df.iterrows(), total=len(df)):
        img_name = os.path.basename(row["path"])  # get 'xxxx.jpg'
        src_path = os.path.join(src_dir, img_name)
        dst_path = os.path.join(dst_dir, row["broad_class"], img_name)

        if os.path.exists(src_path):
            shutil.copy(src_path, dst_path)

    print(f"✅ {split_name} subset ready: {dst_dir}")


# -------------------------
# Main Function
# -------------------------
def main():
    print("🚀 Preparing DeepFashion2 subset...")
    print(f"Reading train.csv: {TRAIN_CSV}")
    print(f"Reading validation.csv: {VAL_CSV}")

    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VAL_CSV)

    # Debug: Show few rows
    print("\n📊 Sample from train.csv:")
    print(train_df.head(3)[["path", "category_name"]])

    train_out = os.path.join(OUTPUT_DIR, "train")
    val_out = os.path.join(OUTPUT_DIR, "val")

    copy_split(train_df, TRAIN_IMG_DIR, train_out, TRAIN_SAMPLE, "train")
    copy_split(val_df, VAL_IMG_DIR, val_out, VAL_SAMPLE, "validation")

    print("\n✅ Dataset subset created successfully!")
    print(f"Saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
