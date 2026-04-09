"""
train.py — Full Training Pipeline for Laptop Price Prediction
=============================================================
Dataset: Real-world laptop dataset (1303 laptops, prices in Euros)
Source : Kaggle — Laptop Price by Muhammet Varlı

Run:
    python project_root/training/train.py
"""

import os, re, pickle, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(BASE_DIR, "data", "laptops.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "laptop_price_model.pkl")
ENC_PATH   = os.path.join(BASE_DIR, "..", "columns_label_encodings.pkl")


def engineer_features(df):
    print("\n⚙️  Feature Engineering...")
    df = df.drop(columns=['laptop_ID', 'Product'], errors='ignore')

    df['Ram_GB']    = df['Ram'].str.replace('GB', '').astype(int)
    df['Weight_KG'] = df['Weight'].str.replace('kg', '').astype(float)
    df = df.drop(columns=['Ram', 'Weight'])

    def total_storage(mem):
        total = 0
        for p in str(mem).replace('Hybrid','SSD').upper().split('+'):
            try:
                val = float(re.search(r'[\d\.]+', p).group())
                if 'TB' in p: val *= 1024
                total += val
            except: pass
        return total

    df['Storage_GB'] = df['Memory'].apply(total_storage)
    df['SSD']        = df['Memory'].apply(lambda x: 1 if 'SSD' in str(x).upper() or 'FLASH' in str(x).upper() else 0)
    df = df.drop(columns=['Memory'])

    df['CPU_Brand'] = df['Cpu'].apply(lambda x: 'Intel' if 'Intel' in x else ('AMD' if 'AMD' in x else 'Other'))
    df['GPU_Brand'] = df['Gpu'].apply(lambda x: 'Nvidia' if 'Nvidia' in x else ('Intel' if 'Intel' in x else ('AMD' if 'AMD' in x else 'Other')))
    df = df.drop(columns=['Cpu', 'Gpu'])

    def get_pixels(res):
        m = re.search(r'(\d{3,4})x(\d{3,4})', str(res))
        return int(m.group(1)) * int(m.group(2)) if m else 1920*1080

    df['Resolution_MP'] = df['ScreenResolution'].apply(get_pixels)
    df['IPS']           = df['ScreenResolution'].apply(lambda x: 1 if 'IPS' in str(x) else 0)
    df['Touchscreen']   = df['ScreenResolution'].apply(lambda x: 1 if 'Touch' in str(x) else 0)
    df = df.drop(columns=['ScreenResolution'])

    def simplify_os(os):
        os = str(os).lower()
        if 'windows' in os: return 'Windows'
        if 'mac' in os:     return 'macOS'
        if 'linux' in os:   return 'Linux'
        if 'chrome' in os:  return 'Chrome OS'
        return 'Other'
    df['OpSys'] = df['OpSys'].apply(simplify_os)

    print(f"   Final features: {df.columns.tolist()}")
    return df


def preprocess(df):
    print("\n🔢 Encoding categorical features...")
    label_encoders = {}
    for col in ['Company', 'TypeName', 'OpSys', 'CPU_Brand', 'GPU_Brand']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
        print(f"   {col}: {list(le.classes_)}")

    with open(ENC_PATH, "wb") as f:
        pickle.dump(label_encoders, f)

    return df.drop(columns=['Price_euros']), df['Price_euros'], label_encoders


def train_models(X_train, X_test, y_train, y_test):
    print("\n🤖 Training 5 models...")
    models = {
        'Linear Regression' : LinearRegression(),
        'Ridge Regression'  : Ridge(alpha=10),
        'Decision Tree'     : DecisionTreeRegressor(max_depth=8, random_state=42),
        'Random Forest'     : RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'Gradient Boosting' : GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=4, random_state=42),
    }
    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mae  = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2   = r2_score(y_test, y_pred)
        results.append({'Model': name, 'MAE': round(mae,2), 'RMSE': round(rmse,2), 'R2': round(r2,4), 'object': model})
        print(f"   {name:<25} | R²={r2:.4f} | MAE=€{mae:.2f}")

    return pd.DataFrame(results).sort_values('R2', ascending=False).reset_index(drop=True), models


def save_model(results_df, models, label_encoders, feature_names):
    best_name  = results_df.iloc[0]['Model']
    best_model = models[best_name]
    artifact   = {
        'model': best_model, 'label_encoders': label_encoders,
        'feature_names': feature_names, 'model_name': best_name,
        'r2': results_df.iloc[0]['R2'], 'mae': results_df.iloc[0]['MAE'],
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(artifact, f)
    print(f"\n✅ Best: {best_name} (R²={results_df.iloc[0]['R2']}, MAE=€{results_df.iloc[0]['MAE']})")
    return best_model, best_name


if __name__ == "__main__":
    print("=" * 60)
    print("  💻 Laptop Price Prediction — Training Pipeline")
    print("  Dataset: 1303 real laptops | Prices in Euros")
    print("=" * 60)

    df                               = pd.read_csv(DATA_PATH, encoding='latin1')
    print(f"\n📂 Loaded: {df.shape[0]} rows × {df.shape[1]} columns")
    df                               = engineer_features(df)
    X, y, label_encoders             = preprocess(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    results_df, models = train_models(X_train, X_test, y_train, y_test)
    print("\n🏆 Leaderboard:")
    print(results_df[['Model','R2','MAE','RMSE']].to_string(index=False))

    save_model(results_df, models, label_encoders, list(X.columns))
    print("\n🎉 Training complete!")
