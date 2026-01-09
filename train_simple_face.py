"""
Train SIMPLE Random Forest with only 5 basic features
For easy integration in app_production.py
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
import cv2
from scipy.fft import fft2, fftshift
from frequency import azimuthalAverage
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
from tqdm import tqdm

FAKE_PATH = r"/path/to/dataset/training_fake"  # TODO: Set your path (see SETUP_LOCAL.md)
REAL_PATH = r"/path/to/dataset/training_real"  # TODO: Set your path (see SETUP_LOCAL.md)

def extract_simple_features(img_path):
    """Extract only 5 features that match app_production.py"""
    try:
        img = Image.open(img_path).convert('RGB')
        img_array = np.array(img)
        
        gray = np.mean(img_array, axis=2).astype(np.float64)
        window = np.hanning(gray.shape[0])[:, None] * np.hanning(gray.shape[1])[None, :]
        gray_windowed = gray * window
        
        fft_result = fft2(gray_windowed)
        fft_shifted = fftshift(fft_result)
        
        psd_2d = np.abs(fft_shifted) ** 2
        radial_profile = azimuthalAverage(psd_2d)
        psd1D = 10 * np.log10(radial_profile + 1e-10)
        
        psd_len = len(psd1D)
        
        if psd_len < 50:
            return None
        
        features = {
            'tail_70': np.gradient(psd1D)[int(0.7*psd_len):].mean(),
            'tail_80': np.gradient(psd1D)[int(0.8*psd_len):].mean(),
            'tail_90': np.gradient(psd1D)[int(0.9*psd_len):].mean(),
            'hf_lf_ratio': np.mean(psd1D[int(0.7*psd_len):]) / (np.mean(psd1D[:int(0.4*psd_len)]) + 1e-10),
            'std_power': np.std(psd1D)
        }
        
        return features
        
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None

print("=" * 60)
print("Training SIMPLE Random Forest (5 features only)")
print("=" * 60)

results = []

# Process FAKE
print("\nProcessing FAKE images...")
fake_files = [f for f in os.listdir(FAKE_PATH) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
for filename in tqdm(fake_files):
    img_path = os.path.join(FAKE_PATH, filename)
    features = extract_simple_features(img_path)
    if features:
        features['label'] = 1  # FAKE = 1
        results.append(features)

# Process REAL
print("\nProcessing REAL images...")
real_files = [f for f in os.listdir(REAL_PATH) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
for filename in tqdm(real_files):
    img_path = os.path.join(REAL_PATH, filename)
    features = extract_simple_features(img_path)
    if features:
        features['label'] = 0  # REAL = 0
        results.append(features)

df = pd.DataFrame(results)

print(f"\nDataset size: {len(df)} images")
print(f"FAKE: {(df['label'] == 1).sum()}, REAL: {(df['label'] == 0).sum()}")

# Split data
X = df.drop('label', axis=1)
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")

# Train Random Forest
print("\nTraining Random Forest...")
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=20,
    min_samples_leaf=10,
    random_state=42,
    class_weight='balanced'
)

rf.fit(X_train, y_train)

# Evaluate
y_pred_test = rf.predict(X_test)
test_acc = accuracy_score(y_test, y_pred_test)

y_pred_train = rf.predict(X_train)
train_acc = accuracy_score(y_train, y_pred_train)

print(f"\n✅ Training complete!")
print(f"Test Accuracy: {test_acc*100:.1f}%")
print(f"Train Accuracy: {train_acc*100:.1f}%")

print("\nClassification Report (Test Set):")
print(classification_report(y_test, y_pred_test, target_names=['REAL', 'FAKE']))

print("\nConfusion Matrix (Test Set):")
cm = confusion_matrix(y_test, y_pred_test)
print(f"              Predicted")
print(f"         REAL    FAKE")
print(f"Actual REAL {cm[0,0]:4d}    {cm[0,1]:4d}")
print(f"       FAKE {cm[1,0]:4d}    {cm[1,1]:4d}")

# Feature importance
feature_names = X.columns.tolist()
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

print("\nFeature Importance:")
for i in range(len(feature_names)):
    print(f"  {feature_names[indices[i]]}: {importances[indices[i]]:.3f}")

# Save model
model_path = 'face_rf_simple.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(rf, f)

print(f"\n✅ Model saved to: {model_path}")

# Test on one real image
print("\n" + "="*60)
print("Quick Test on Real Image")
print("="*60)

test_img = os.path.join(REAL_PATH, real_files[0])
features_test = extract_simple_features(test_img)

if features_test:
    X_test_single = np.array([[
        features_test['tail_70'],
        features_test['tail_80'],
        features_test['tail_90'],
        features_test['hf_lf_ratio'],
        features_test['std_power']
    ]])
    
    proba = rf.predict_proba(X_test_single)[0]
    ai_score = proba[1] * 100
    
    print(f"\nImage: {real_files[0]}")
    print(f"  P(REAL): {proba[0]*100:.1f}%")
    print(f"  P(FAKE): {proba[1]*100:.1f}%")
    print(f"  AI Score: {ai_score:.1f}%")
    print(f"  Verdict: {'AI-GENERATED' if ai_score > 60 else 'REAL'}")
