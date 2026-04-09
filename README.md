# 💻 Laptop Price Prediction

End-to-end Machine Learning project predicting laptop prices using a real-world dataset of 1303 laptops, with feature engineering, model comparison, Flask REST API, and Docker deployment.

---

## 📊 Dataset
- **Source**: [Kaggle — Laptop Price by Muhammet Varlı](https://www.kaggle.com/datasets/muhammetvarol/laptop-prices-dataset)
- **Records**: 1303 real laptops
- **Target**: `Price_euros` (€174 – €6099)
- **Raw features**: Company, CPU, RAM, Memory, GPU, Screen, OS, Weight

---

## 📁 Project Structure

```
Laptop_Price_Prediction/
│
├── src.ipynb                          ← Main notebook (EDA + training + evaluation)
├── Dockerfile                         ← Docker image for the API
├── requirements.txt                   ← Python dependencies
├── columns_label_encodings.pkl        ← Saved label encoders
├── README.md
│
├── dockerfiles/
│   └── Dockerfile.train               ← Docker image for training only
│
└── project_root/
    ├── data/
    │   └── laptops.csv                ← Raw dataset (1303 rows)
    ├── models/
    │   └── laptop_price_model.pkl     ← Trained model artifact
    ├── training/
    │   └── train.py                   ← Full training pipeline
    └── api/
        └── app.py                     ← Flask REST API
```

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the model
python project_root/training/train.py

# 3. Start the API
python project_root/api/app.py

# 4. Predict
curl -X POST http://localhost:5000/predict \
     -H "Content-Type: application/json" \
     -d '{
           "Company": "Dell", "TypeName": "Notebook",
           "Inches": 15.6, "Ram_GB": 8, "Storage_GB": 256,
           "SSD": 1, "Weight_KG": 2.0, "OpSys": "Windows",
           "CPU_Brand": "Intel", "GPU_Brand": "Nvidia",
           "Resolution_MP": 2073600, "IPS": 1, "Touchscreen": 0
         }'
```

---

## 🐳 Docker

```bash
docker build -t laptop-price-api .
docker run -p 5000:5000 laptop-price-api
```

---

## 🤖 Model Results

| Model              | R² Score | MAE (€) |
|--------------------|----------|---------|
| **Random Forest**  | **0.7747**| **€203** |
| Gradient Boosting  | 0.7745   | €194    |
| Decision Tree      | 0.6466   | €265    |
| Ridge Regression   | 0.6639   | €299    |
| Linear Regression  | 0.6633   | €300    |

---

## ⚙️ Feature Engineering

| Raw Feature      | Engineered Feature    | Description                        |
|------------------|-----------------------|------------------------------------|
| Ram              | `Ram_GB`              | Numeric: 4, 8, 16, 32, 64          |
| Weight           | `Weight_KG`           | Numeric float                      |
| Memory           | `Storage_GB`, `SSD`   | Total GB + SSD flag (0/1)          |
| Cpu              | `CPU_Brand`           | Intel / AMD / Other                |
| Gpu              | `GPU_Brand`           | Nvidia / Intel / AMD / Other       |
| ScreenResolution | `Resolution_MP`, `IPS`, `Touchscreen` | Pixels + flags      |
| OpSys            | `OpSys`               | Windows / macOS / Linux / Other    |
