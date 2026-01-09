"""
Calibrare avansata cu features 2D spectrum pentru FACE dataset
"""

import os
import numpy as np
from PIL import Image
import cv2
from scipy.fft import fft2, fftshift
from frequency import azimuthalAverage
import pandas as pd
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import pickle

DATASET_PATH = r/path/to/dataset
FAKE_PATH = os.path.join(DATASET_PATH, "training_fake")
REAL_PATH = os.path.join(DATASET_PATH, "training_real")

def compute_advanced_features(img_path):
    """Features FFT + 2D spectrum features"""
    try:
        img = Image.open(img_path).convert('RGB')
        img_array = np.array(img)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        f = fft2(gray)
        fshift = fftshift(f)
        magnitude_spectrum = np.abs(fshift)
        magnitude_spectrum_log = np.log1p(magnitude_spectrum)
        
        psd1D = azimuthalAverage(magnitude_spectrum_log)
        
        if psd1D is None or len(psd1D) < 50:
            return None
        
        features = {}
        psd_len = len(psd1D)
        
        # Radial PSD features
        for pct in [0.6, 0.7, 0.8, 0.9]:
            idx_start = int(pct * psd_len)
            idx_end = min(idx_start + 10, psd_len)
            
            if idx_end > idx_start + 1:
                x = np.arange(idx_start, idx_end)
                y = psd1D[idx_start:idx_end]
                valid = np.isfinite(y)
                
                if np.sum(valid) > 1:
                    coeffs = np.polyfit(x[valid], y[valid], 1)
                    features[f'tail_{int(pct*100)}'] = coeffs[0]
                else:
                    features[f'tail_{int(pct*100)}'] = 0
            else:
                features[f'tail_{int(pct*100)}'] = 0
        
        # Power features
        cutoff = len(psd1D) // 2
        features['hf_lf_ratio'] = np.mean(psd1D[cutoff:]) / np.mean(psd1D[:cutoff])
        features['mean_power'] = np.mean(psd1D)
        features['std_power'] = np.std(psd1D)
        features['power_range'] = np.ptp(psd1D[np.isfinite(psd1D)])
        
        # 2D Spectrum features (ADVANCED)
        h, w = magnitude_spectrum_log.shape
        center_y, center_x = h // 2, w // 2
        
        # Center power
        center_region = magnitude_spectrum_log[center_y-10:center_y+10, center_x-10:center_x+10]
        features['center_power'] = np.mean(center_region)
        
        # Quadrant analysis
        q1 = magnitude_spectrum_log[:center_y, :center_x]
        q2 = magnitude_spectrum_log[:center_y, center_x:]
        q3 = magnitude_spectrum_log[center_y:, :center_x]
        q4 = magnitude_spectrum_log[center_y:, center_x:]
        
        features['q1_mean'] = np.mean(q1)
        features['q2_mean'] = np.mean(q2)
        features['q3_mean'] = np.mean(q3)
        features['q4_mean'] = np.mean(q4)
        
        # Symmetry (AI-generated images often have perfect symmetry)
        features['h_symmetry'] = np.corrcoef(magnitude_spectrum_log[:, :w//2].flatten(), 
                                              np.flip(magnitude_spectrum_log[:, w//2:], axis=1).flatten())[0, 1]
        features['v_symmetry'] = np.corrcoef(magnitude_spectrum_log[:h//2, :].flatten(), 
                                              np.flip(magnitude_spectrum_log[h//2:, :], axis=0).flatten())[0, 1]
        
        # Edge vs center ratio (GAN artifacts)
        edge_region = np.concatenate([
            magnitude_spectrum_log[0, :],
            magnitude_spectrum_log[-1, :],
            magnitude_spectrum_log[:, 0],
            magnitude_spectrum_log[:, -1]
        ])
        features['edge_center_ratio'] = np.mean(edge_region) / features['center_power']
        
        # High frequency concentration
        outer_ring = magnitude_spectrum_log.copy()
        outer_ring[center_y-50:center_y+50, center_x-50:center_x+50] = 0
        features['outer_power'] = np.mean(outer_ring[outer_ring > 0])
        
        # Gradient variance (smoothness indicator)
        features['gradient_variance'] = np.var(np.gradient(psd1D))
        
        return features
    
    except Exception as e:
        print(f"Error: {e}")
        return None

def train_rf_classifier():
    """Train Random Forest pe FACE dataset"""
    print("Processing images...")
    
    results = []
    
    # FAKE
    print("\nFAKE images...")
    fake_files = [f for f in os.listdir(FAKE_PATH) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    for filename in tqdm(fake_files[:960]):
        img_path = os.path.join(FAKE_PATH, filename)
        features = compute_advanced_features(img_path)
        if features:
            features['label'] = 1  # FAKE = 1
            results.append(features)
    
    # REAL
    print("\nREAL images...")
    real_files = [f for f in os.listdir(REAL_PATH) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    for filename in tqdm(real_files[:1081]):
        img_path = os.path.join(REAL_PATH, filename)
        features = compute_advanced_features(img_path)
        if features:
            features['label'] = 0  # REAL = 0
            results.append(features)
    
    df = pd.DataFrame(results)
    df = df.dropna()
    
    print(f"\nTotal: {len(df)} images")
    print(f"FAKE: {len(df[df['label'] == 1])}, REAL: {len(df[df['label'] == 0])}")
    
    # Train/Test split
    X = df.drop('label', axis=1)
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Train Random Forest
    print("\nTraining Random Forest...")
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = rf.predict(X_test)
    print("\n" + "="*60)
    print("RESULTS ON TEST SET (20%)")
    print("="*60)
    print(classification_report(y_test, y_pred, target_names=['REAL', 'FAKE']))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Feature importance
    print("\n" + "="*60)
    print("TOP 10 MOST IMPORTANT FEATURES")
    print("="*60)
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(feature_importance.head(10))
    
    # Save model
    with open('face_rf_model.pkl', 'wb') as f:
        pickle.dump(rf, f)
    print("\nModel saved: face_rf_model.pkl")
    
    # Full dataset accuracy
    y_pred_full = rf.predict(X)
    accuracy = (y == y_pred_full).mean()
    print(f"\nFull dataset accuracy: {accuracy*100:.2f}%")
    
    return rf, feature_importance

if __name__ == "__main__":
    rf, importance = train_rf_classifier()
