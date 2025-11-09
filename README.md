# 👗 AI Fashion Stylist

An **AI-powered fashion recommendation system** that suggests complementary clothing items and styling ideas based on an uploaded image.  
Built using **DeepFashion2**, **CLIP embeddings**, and **ResNet50 classification**, it provides **occasion-aware**, **color-aware**, and **category-smart** outfit recommendations.

---

## 🌟 Features

✅ **Automatic Clothing Detection**
- Detects whether an uploaded image is *topwear, bottomwear, dress,* or *outerwear* using a trained ResNet50 model.

✅ **Smart Outfit Suggestions**
- Suggests **complementary clothing items** (e.g., top → bottomwear, bottom → topwear).  
- Uses **CLIP embeddings** and **FAISS similarity search** for top 5 visually similar items.

✅ **Color Analysis**
- Extracts **dominant colors** using K-Means clustering and provides readable color names.  
- Generates **color harmony tips** and **contrast suggestions**.

✅ **Occasion-Aware Styling**
- Provides fashion tips for *Casual, Formal, Party, Date, Streetwear,* or *Festive* occasions.

✅ **Fully Local Setup**
- Works entirely on your machine once the DeepFashion2 dataset and models are set up.

---

## 🧠 Tech Stack

- **Python 3.10**
- **PyTorch / torchvision**
- **OpenAI CLIP**
- **FAISS (Facebook AI Similarity Search)**
- **Streamlit**
- **scikit-learn / OpenCV / Pillow**
- **DeepFashion2 Dataset**

---

## 🗂️ Project Structure

```
AI-Fashion-Stylist/
│
├── DeepFashion2/
│   ├── deepfashion2_original_images/
│   └── img_info_dataframes/        # train.csv, validation.csv, test.csv
│
├── data_subset/                    # Prepared small subset of DeepFashion2
│
├── models/
│   ├── cloth_classifier_*.pth      # Trained ResNet50 weights
│   ├── fashion_index.faiss         # FAISS index built from CLIP embeddings
│   └── image_paths.pkl             # List of image paths used in FAISS
│
├── scripts/
│   ├── prepare_dataset.py          # Prepares smaller train/val subset
│   ├── train_classifier.py         # Trains clothing classifier
│   ├── build_faiss.py              # Builds CLIP embeddings + FAISS index
│   └── color_utils.py              # Extracts dominant colors
│
├── demo_app.py                     # Streamlit front-end (main app)
└── README.md                       # You're here!
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/<your-username>/AI-Fashion-Stylist.git
cd AI-Fashion-Stylist
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```
> Make sure you're using **Python 3.10** — PyTorch and FAISS may not yet support Python 3.14.

### 3️⃣ Download Dataset
Download the **DeepFashion2 Dataset** from [Kaggle](https://www.kaggle.com/datasets/thusharanair/deepfashion2-original-with-dataframes)  
and extract it inside:
```
AI-Fashion-Stylist/DeepFashion2/
```

---

## 🧩 Run the Complete Pipeline

### 🧱 Step 1 — Prepare Dataset
Create a smaller, manageable subset for local training:
```bash
python scripts/prepare_dataset.py
```

### 🧠 Step 2 — Train Classifier
Train a ResNet50 model to detect clothing type:
```bash
python scripts/train_classifier.py
```

### 🧮 Step 3 — Build FAISS Index
Build CLIP embeddings and similarity index:
```bash
python scripts/build_faiss.py
```

### 💅 Step 4 — Launch the AI Stylist App
Run the Streamlit interface:
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

## 🪄 Example Outputs

| Uploaded Image | Detected | Example Suggestion |
|----------------|-----------|--------------------|
| Black T-shirt | Topwear (dark tone) | “Pair with beige or pastel bottoms and silver accessories.” |
| Blue Jeans | Bottomwear (cool tone) | “Try white or pastel tops with sneakers for a casual vibe.” |
| Red Dress | Dress (vibrant tone, Party) | “Add metallic heels, clutch, and statement jewelry.” |

---

## 📂 Script Details

| File | Description |
|------|--------------|
| `prepare_dataset.py` | Prepares a smaller subset from the DeepFashion2 dataset. |
| `train_classifier.py` | Trains a ResNet50 model to classify clothing items. |
| `build_faiss.py` | Builds CLIP embeddings and stores them in a FAISS index. |
| `color_utils.py` | Extracts dominant colors from an image using k-means. |
| `demo_app.py` | Main Streamlit app — handles image upload, classification, and recommendations. |

---

## 💡 Future Enhancements

- 🤖 **Automatic Occasion Detection** (AI predicts casual/formal/party mode automatically)  
- 🧍 **Virtual Try-On Integration** (overlay clothing on person image)  
- 🛒 **E-commerce Integration** (fetch similar items online)  
- 🎯 **Improved Dual Encoder Retrieval** (better top-bottom pairing)  

---

## 🧾 License

Released under the [MIT License](LICENSE).

---

## 🧑‍💻 Author

**Ritu Pal**  
🎓 B.Tech CSE (AI-ML), Adani University  
📧 [ritupal1626@gmail.com]  
💼 [https://github.com/ritup04]

**Helly Khambhatwala**  
🎓 B.Tech CSE (AI-ML), Adani University  
📧 [helly9328@gmail.com]  
💼 [https://github.com/helly1408]

---

## ⭐ Acknowledgements

- [DeepFashion2 Dataset](https://github.com/switchablenorms/DeepFashion2)  
- [OpenAI CLIP](https://github.com/openai/CLIP)  
- [FAISS by Meta AI](https://github.com/facebookresearch/faiss)  
- [Streamlit](https://streamlit.io)

---

> 💬 *“Style is a way to say who you are without having to speak.”*  
> — *Rachel Zoe*
