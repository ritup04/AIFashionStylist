# 👗 AI Fashion Stylist

An **AI-powered fashion recommendation system** that suggests complementary clothing items and styling ideas based on an uploaded image.  
Built using **DeepFashion2 (Kaggle version)**, **CLIP embeddings**, and **ResNet50 classification**, it provides **occasion-aware**, **color-aware**, and **category-smart** outfit recommendations.

---

## 🌟 Features

✅ **Automatic Clothing Detection**
- Detects whether an uploaded image is *Topwear*, *Bottomwear*, *Dress*, or *Outerwear* using a trained ResNet50 model.

✅ **Smart Outfit Suggestions**
- Suggests **complementary clothing items** (e.g., Top → Bottomwear, Bottom → Topwear).  
- Uses **CLIP embeddings** and **FAISS similarity search** for top 5 visually similar items.

✅ **Color Analysis**
- Extracts **dominant colors** using K-Means clustering and provides human-readable color names.  
- Generates **color harmony tips** and **contrast-based fashion advice**.

✅ **Occasion-Aware Styling**
- Provides dynamic styling tips based on detected color tone and outfit type — for *Casual, Formal, Party, Streetwear,* or *Festive* occasions.

✅ **Local Execution**
- Works fully offline once the dataset and models are set up on your local machine.

---

## 🧠 Tech Stack

- **Python 3.10**
- **PyTorch / torchvision**
- **OpenAI CLIP**
- **FAISS (Facebook AI Similarity Search)**
- **Streamlit**
- **scikit-learn / OpenCV / Pillow**
- **Kaggle: DeepFashion2 Original Dataset**

---

## 🗂️ Project Structure

```
AI-Fashion-Stylist/
│
├── DeepFashion2/
│   ├── deepfashion2_original_images/
│   └── img_info_dataframes/        # train.csv, validation.csv, test.csv
│
├── data_subset/                    # Prepared smaller dataset for training
│
├── models/
│   ├── cloth_classifier.pth        # Trained ResNet50 model
│   ├── fashion_index.faiss         # FAISS index built from CLIP embeddings
│   └── image_paths.pkl             # List of images used in FAISS index
│
├── scripts/
│   ├── prepare_dataset.py          # Dataset pre-processing
│   ├── train_classifier.py         # Model training
│   ├── build_faiss.py              # CLIP + FAISS builder
│   └── color_utils.py              # Color extraction utility
│
├── demo_app.py                     # Streamlit application
└── README.md                       # You're reading it!
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/ritup04/AI-Fashion-Stylist.git
cd AI-Fashion-Stylist
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```
> ⚠️ Make sure you’re using **Python 3.10** for best compatibility with PyTorch and FAISS.

---

## 🧩 Dataset

This project uses the **DeepFashion2 dataset** available on Kaggle.

📦 **Dataset Link:**  
[DeepFashion2 Original with Dataframes (Kaggle)](https://www.kaggle.com/datasets/thusharanair/deepfashion2-original-with-dataframes)

After downloading, extract it into:
```
AI-Fashion-Stylist/DeepFashion2/
```

> ⚠️ The dataset is **not included in this repository** due to its large size.

---

## 🧩 Run the Complete Pipeline

### 🧱 Step 1 — Prepare Dataset
```bash
python scripts/prepare_dataset.py
```

### 🧠 Step 2 — Train Classifier
```bash
python scripts/train_classifier.py
```

### 🧮 Step 3 — Build FAISS Index
```bash
python scripts/build_faiss.py
```

### 💅 Step 4 — Launch the AI Stylist App
```bash
streamlit run demo_app.py
```

---

## 🖼️ Demo Workflow

| Step | Description |
|------|--------------|
| 🖼️ Upload | Upload an outfit image (topwear, bottomwear, dress, etc.) |
| 🧠 Detection | The model classifies the outfit type |
| 🎨 Color Extraction | Extracts dominant colors & harmony |
| 🛍️ Similar Items | Displays top 5 similar or complementary outfits |
| 💡 Styling Tips | Gives dynamic, occasion-based fashion advice |

---

## 🧾 Authors

👩‍💻 **Ritu Pal**  
🎓 B.Tech CSE (AI-ML), Adani University  
📧 ritupal1626@gmail.com  
💼 [GitHub: ritup04](https://github.com/ritup04)

👩‍💻 **Helly Khambhatwala**  
🎓 B.Tech CSE (AI-ML), Adani University  
📧 helly9328@gmail.com  
💼 [GitHub: helly1408](https://github.com/helly1408)
