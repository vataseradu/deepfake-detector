"""
Advanced Hybrid Detection System - Master's Thesis
==================================================
Combines multiple feature extraction methods with Machine Learning

Features:
1. FFT Spectral Analysis (Log Ratio, Tail Gradients)
2. Wavelet Analysis (Daubechies - detect high-freq artifacts)
3. Local Binary Patterns (texture inconsistencies)
4. Gradient Magnitude Distribution (edge sharpness)
5. ELA (compression artifacts)

Model: Random Forest Classifier with Cross-Validation
"""

import os
import glob
import csv
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageChops, ImageFilter
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import warnings
warnings.filterwarnings('ignore')

try:
    import pywt  # Wavelet transform
except ImportError:
    print("⚠️ Installing pywt (PyWavelets)...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'PyWavelets'])
    import pywt

from skimage.feature import local_binary_pattern
from scipy.stats import entropy, skew, kurtosis

# === FEATURE EXTRACTION ===

def extract_wavelet_features(img_array):
    """Wavelet decomposition - detectează artefacte AI în high-freq"""
    try:
        # Daubechies wavelet (db4) - bun pentru imagini
        coeffs = pywt.dwt2(img_array, 'db4')
        cA, (cH, cV, cD) = coeffs  # Approximation + Details (H, V, Diagonal)
        
        # Features din componentele high-frequency
        features = {
            'wavelet_cH_std': float(np.std(cH)),
            'wavelet_cV_std': float(np.std(cV)),
            'wavelet_cD_std': float(np.std(cD)),
            'wavelet_cH_energy': float(np.sum(cH**2)),
            'wavelet_detail_ratio': float(np.sum(cH**2) / (np.sum(cA**2) + 1e-10))
        }
        return features
    except:
        return {
            'wavelet_cH_std': 0.0,
            'wavelet_cV_std': 0.0,
            'wavelet_cD_std': 0.0,
            'wavelet_cH_energy': 0.0,
            'wavelet_detail_ratio': 0.0
        }

def extract_lbp_features(img_array):
    """Local Binary Patterns - detectează inconsistențe în texturi"""
    try:
        # LBP cu radius=3, points=24 (mai detaliat)
        lbp = local_binary_pattern(img_array, P=24, R=3, method='uniform')
        
        # Histogram și statistici
        hist, _ = np.histogram(lbp.ravel(), bins=26, range=(0, 26), density=True)
        
        features = {
            'lbp_entropy': float(entropy(hist + 1e-10)),
            'lbp_uniformity': float(np.sum(hist**2)),
            'lbp_mean': float(np.mean(lbp)),
            'lbp_std': float(np.std(lbp))
        }
        return features
    except:
        return {
            'lbp_entropy': 0.0,
            'lbp_uniformity': 0.0,
            'lbp_mean': 0.0,
            'lbp_std': 0.0
        }

def extract_gradient_features(img_array):
    """Gradient Magnitude Distribution - edges sharpness"""
    try:
        # Sobel gradients
        gx = np.gradient(img_array, axis=1)
        gy = np.gradient(img_array, axis=0)
        magnitude = np.sqrt(gx**2 + gy**2)
        
        features = {
            'gradient_mean': float(np.mean(magnitude)),
            'gradient_std': float(np.std(magnitude)),
            'gradient_max': float(np.max(magnitude)),
            'gradient_skew': float(skew(magnitude.ravel())),
            'gradient_kurtosis': float(kurtosis(magnitude.ravel()))
        }
        return features
    except:
        return {
            'gradient_mean': 0.0,
            'gradient_std': 0.0,
            'gradient_max': 0.0,
            'gradient_skew': 0.0,
            'gradient_kurtosis': 0.0
        }

def analyze_image(file_info):
    """Extrage TOATE features dintr-o imagine"""
    img_path, label = file_info
    
    try:
        with Image.open(img_path) as im:
            im = im.convert('RGB')
            
            features = {
                "type": label,
                "filename": os.path.basename(img_path)
            }
            
            # --- 1. ELA ---
            pid = multiprocessing.current_process().pid
            temp_filename = f"temp_hybrid_{pid}.jpg"
            
            im.save(temp_filename, 'JPEG', quality=90)
            resaved = Image.open(temp_filename)
            ela_im = ImageChops.difference(im, resaved)
            
            ela_array = np.array(ela_im)
            gray_ela = np.mean(ela_array, axis=2)
            features['ela_std'] = float(np.std(gray_ela))
            features['ela_mean'] = float(np.mean(gray_ela))
            
            try:
                resaved.close()
                if os.path.exists(temp_filename):
                    os.remove(temp_filename)
            except:
                pass

            # --- 2. FFT Analysis ---
            img_gray = np.array(im.convert('L'))
            f = np.fft.fft2(img_gray)
            fshift = np.fft.fftshift(f)
            magnitude = np.abs(fshift) ** 2
            
            # Azimuthal Average
            y, x = np.indices(magnitude.shape)
            center = np.array([(x.max()-x.min())/2.0, (y.max()-y.min())/2.0])
            r = np.hypot(x - center[0], y - center[1])
            ind = np.argsort(r.flat)
            r_sorted = r.flat[ind]
            i_sorted = magnitude.flat[ind]
            r_int = r_sorted.astype(int)
            tbin = np.bincount(r_int, i_sorted)
            nr = np.bincount(r_int)
            nr[nr == 0] = 1
            psd1D = tbin / nr
            
            psd1D = psd1D[5:]
            valid = psd1D > 0
            if np.sum(valid) < 50:
                return None
            psd1D = psd1D[valid]
            n_freq = len(psd1D)
            
            # HF Ratio
            cutoff = int(0.6 * n_freq)
            low_p = np.mean(psd1D[:cutoff])
            high_p = np.mean(psd1D[cutoff:])
            
            features['log_hf_ratio'] = -10.0
            if low_p > 0 and high_p > 0:
                features['log_hf_ratio'] = np.log10(high_p / low_p)
            
            # Multiple Tail Gradients
            for zone, start_pct in [('70', 0.7), ('80', 0.8), ('90', 0.9)]:
                start_idx = int(start_pct * n_freq)
                if len(psd1D[start_idx:]) > 5:
                    x_tail = np.log10(np.arange(start_idx, n_freq) + 1).reshape(-1, 1)
                    y_tail = np.log10(psd1D[start_idx:])
                    model = LinearRegression().fit(x_tail, y_tail)
                    features[f'tail_{zone}'] = float(model.coef_[0])
                else:
                    features[f'tail_{zone}'] = 0.0
            
            # Overall Spectral Slope
            start_20 = int(0.2 * n_freq)
            if n_freq - start_20 > 20:
                x_full = np.log10(np.arange(start_20, n_freq) + 1).reshape(-1, 1)
                y_full = np.log10(psd1D[start_20:])
                model_full = LinearRegression().fit(x_full, y_full)
                features['spectral_slope'] = float(model_full.coef_[0])
            else:
                features['spectral_slope'] = 0.0

            # --- 3. Wavelet Analysis ---
            wavelet_feats = extract_wavelet_features(img_gray)
            features.update(wavelet_feats)
            
            # --- 4. LBP Analysis ---
            lbp_feats = extract_lbp_features(img_gray)
            features.update(lbp_feats)
            
            # --- 5. Gradient Analysis ---
            grad_feats = extract_gradient_features(img_gray)
            features.update(grad_feats)

            return features
            
    except Exception as e:
        return None

def main():
    # === CONFIGURARE ===
    BASE_PATH = r/path/to/dataset
    REAL_PATH = os.path.join(BASE_PATH, "training_real") 
    FAKE_PATH = os.path.join(BASE_PATH, "training_fake")
    
    print("🚀 ADVANCED HYBRID DETECTION SYSTEM")
    print("="*70)
    print(f"📂 Dataset: {BASE_PATH}")
    
    if not os.path.exists(REAL_PATH):
        print(f"❌ Eroare: {REAL_PATH} nu există")
        return

    # Indexare
    real_files = [(f, "REAL") for f in glob.glob(os.path.join(REAL_PATH, "*.*"))]
    fake_files = [(f, "FAKE") for f in glob.glob(os.path.join(FAKE_PATH, "*.*"))]
    
    print(f"   Găsite: {len(real_files)} REALE, {len(fake_files)} FAKE")

    # Sampling
    SAMPLE_SIZE = 1000
    if len(real_files) > SAMPLE_SIZE:
        print(f"⚡ Analizăm {SAMPLE_SIZE} imagini per clasă (total {SAMPLE_SIZE*2})")
        random.shuffle(real_files)
        random.shuffle(fake_files)
        real_files = real_files[:SAMPLE_SIZE]
        fake_files = fake_files[:SAMPLE_SIZE]
    
    all_files = real_files + fake_files
    
    # Procesare
    print(f"\n🔬 Extracție features (FFT + Wavelet + LBP + Gradient + ELA)...")
    results = []
    with ProcessPoolExecutor() as executor:
        results_gen = list(tqdm(executor.map(analyze_image, all_files), total=len(all_files)))
    
    results = [r for r in results_gen if r is not None]
    
    if len(results) < 100:
        print("❌ Eroare: Prea puține imagini procesate")
        return

    print(f"✅ Procesate: {len(results)} imagini")
    
    # Salvare CSV
    csv_file = "advanced_features.csv"
    keys = results[0].keys()
    with open(csv_file, 'w', newline='') as f:
        dict_writer = csv.DictWriter(f, keys)
        dict_writer.writeheader()
        dict_writer.writerows(results)
    print(f"💾 Features salvate: {csv_file}")

    # === MACHINE LEARNING ===
    print("\n" + "="*70)
    print("🤖 RANDOM FOREST CLASSIFIER")
    print("="*70)
    
    # Prepare data
    feature_names = [k for k in results[0].keys() if k not in ['type', 'filename']]
    X = np.array([[r[f] for f in feature_names] for r in results])
    y = np.array([1 if r['type'] == 'FAKE' else 0 for r in results])
    
    print(f"📊 Features: {len(feature_names)}")
    print(f"   Samples: {len(X)} (REAL={np.sum(y==0)}, FAKE={np.sum(y==1)})")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    
    # Train Random Forest
    print("\n🌲 Antrenare Random Forest...")
    rf = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    # Predictions
    y_pred = rf.predict(X_test)
    y_proba = rf.predict_proba(X_test)[:, 1]
    
    # Evaluation
    print("\n" + "="*70)
    print("📈 REZULTATE FINALE")
    print("="*70)
    print(classification_report(y_test, y_pred, target_names=['REAL', 'FAKE']))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(f"   TN={cm[0,0]} | FP={cm[0,1]}")
    print(f"   FN={cm[1,0]} | TP={cm[1,1]}")
    
    # ROC AUC
    auc = roc_auc_score(y_test, y_proba)
    print(f"\n🎯 ROC AUC Score: {auc:.4f}")
    
    # Cross-validation
    print("\n🔄 Cross-Validation (5-fold)...")
    cv_scores = cross_val_score(rf, X, y, cv=5, scoring='accuracy', n_jobs=-1)
    print(f"   Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # Feature Importance
    print("\n" + "="*70)
    print("🔍 TOP 15 MOST IMPORTANT FEATURES")
    print("="*70)
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    for i in range(min(15, len(feature_names))):
        idx = indices[i]
        print(f"{i+1:2d}. {feature_names[idx]:25s} : {importances[idx]:.4f}")
    
    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Advanced Hybrid Detection - ML Analysis', fontsize=16, fontweight='bold')
    
    # 1. Feature Importance
    top_n = 15
    top_indices = indices[:top_n]
    axes[0, 0].barh(range(top_n), importances[top_indices], color='steelblue')
    axes[0, 0].set_yticks(range(top_n))
    axes[0, 0].set_yticklabels([feature_names[i] for i in top_indices])
    axes[0, 0].set_xlabel('Importance')
    axes[0, 0].set_title('Top 15 Features')
    axes[0, 0].invert_yaxis()
    
    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    axes[0, 1].plot(fpr, tpr, linewidth=2, label=f'ROC (AUC={auc:.4f})')
    axes[0, 1].plot([0, 1], [0, 1], 'k--', label='Random')
    axes[0, 1].set_xlabel('False Positive Rate')
    axes[0, 1].set_ylabel('True Positive Rate')
    axes[0, 1].set_title('ROC Curve')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Confusion Matrix Heatmap
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    im = axes[1, 0].imshow(cm_normalized, cmap='Blues', aspect='auto')
    axes[1, 0].set_xticks([0, 1])
    axes[1, 0].set_yticks([0, 1])
    axes[1, 0].set_xticklabels(['REAL', 'FAKE'])
    axes[1, 0].set_yticklabels(['REAL', 'FAKE'])
    axes[1, 0].set_xlabel('Predicted')
    axes[1, 0].set_ylabel('Actual')
    axes[1, 0].set_title('Confusion Matrix (Normalized)')
    
    # Add text annotations
    for i in range(2):
        for j in range(2):
            text = axes[1, 0].text(j, i, f'{cm_normalized[i, j]:.2f}\n({cm[i, j]})',
                                  ha="center", va="center", color="black")
    
    # 4. Distribution of Top Feature
    top_feature = feature_names[indices[0]]
    real_vals = [r[top_feature] for r in results if r['type'] == 'REAL']
    fake_vals = [r[top_feature] for r in results if r['type'] == 'FAKE']
    axes[1, 1].hist(real_vals, bins=30, alpha=0.6, label='REAL', color='green')
    axes[1, 1].hist(fake_vals, bins=30, alpha=0.6, label='FAKE', color='red')
    axes[1, 1].set_title(f'Distribution: {top_feature}')
    axes[1, 1].set_xlabel('Value')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig("advanced_hybrid_results.png", dpi=150)
    print("\n📊 Grafic salvat: advanced_hybrid_results.png")
    print("="*70)
    plt.show()

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
