"""
RETRAINING SCRIPT - Kaggle Art Dataset
========================================
Dataset: AI Generated Images vs Real Images (Art/Photos)
- REAL: 434 imagini (C:/Users/Vatase Radu/Downloads/DATASET_antrenare/RealArt/)
- FAKE: 536 imagini (C:/Users/Vatase Radu/Downloads/DATASET_antrenare/AiArtData/)
- Total: 970 imagini (art & photos diverse)
"""

import os
import glob
import pickle
import numpy as np
from PIL import Image, ImageChops
import pywt
from skimage.feature import local_binary_pattern
from scipy.stats import entropy
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime

# Feature extraction (same as app)
def extract_features_from_path(img_path, label):
    """Extract all forensic features from image path"""
    try:
        img = Image.open(img_path).convert('RGB')
        
        # === 1. ELA (Error Level Analysis) ===
        temp_file = "temp_ela_train.jpg"
        img.save(temp_file, 'JPEG', quality=90)
        resaved = Image.open(temp_file)
        ela_im = ImageChops.difference(img, resaved)
        ela_array = np.array(ela_im)
        gray_ela = np.mean(ela_array, axis=2)
        
        ela_std = float(np.std(gray_ela))
        ela_mean = float(np.mean(gray_ela))
        ela_max = float(np.max(gray_ela))
        
        resaved.close()
        try:
            os.remove(temp_file)
        except:
            pass
        
        # === 2. FFT Analysis ===
        img_gray = np.array(img.convert('L'))
        f = np.fft.fft2(img_gray)
        fshift = np.fft.fftshift(f)
        magnitude = np.abs(fshift) ** 2
        
        # Azimuthal average for PSD
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
        
        # Trim and validate
        psd1D = psd1D[5:]
        valid = psd1D > 0
        if np.sum(valid) < 50:
            return None
        psd1D = psd1D[valid]
        n_freq = len(psd1D)
        
        # HF Ratio
        cutoff = int(0.7 * n_freq)
        low_p = np.mean(psd1D[:cutoff])
        high_p = np.mean(psd1D[cutoff:])
        log_hf_ratio = np.log10(high_p / low_p) if low_p > 0 and high_p > 0 else -10.0
        
        # Tail gradients
        from sklearn.linear_model import LinearRegression
        
        def compute_tail_gradient(start_pct):
            start_idx = int(start_pct * n_freq)
            if len(psd1D[start_idx:]) < 10:
                return 0.0
            x = np.log10(np.arange(start_idx, n_freq) + 1).reshape(-1, 1)
            y = np.log10(psd1D[start_idx:])
            model = LinearRegression().fit(x, y)
            return float(model.coef_[0])
        
        tail_70 = compute_tail_gradient(0.7)
        tail_80 = compute_tail_gradient(0.8)
        tail_90 = compute_tail_gradient(0.9)
        
        # === 3. Wavelet Transform ===
        coeffs = pywt.dwt2(img_gray, 'db1')
        cA, (cH, cV, cD) = coeffs
        
        wavelet_cH_std = float(np.std(cH))
        wavelet_cV_std = float(np.std(cV))
        wavelet_cD_std = float(np.std(cD))
        wavelet_energy = float(np.sum(cH**2) + np.sum(cV**2) + np.sum(cD**2))
        
        # === 4. LBP (Local Binary Patterns) ===
        lbp = local_binary_pattern(img_gray, P=8, R=1, method='uniform')
        hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
        hist = hist.astype(float) / hist.sum()
        lbp_entropy = entropy(hist + 1e-10)
        lbp_std = float(np.std(lbp))
        
        # === 5. Gradient Analysis ===
        gx = np.gradient(img_gray, axis=1)
        gy = np.gradient(img_gray, axis=0)
        gradient_magnitude = np.sqrt(gx**2 + gy**2)
        
        gradient_mean = float(np.mean(gradient_magnitude))
        gradient_std = float(np.std(gradient_magnitude))
        gradient_max = float(np.max(gradient_magnitude))
        gradient_skew = float(np.mean((gradient_magnitude - gradient_mean)**3) / (gradient_std**3 + 1e-8))
        
        # === 6. Color Statistics ===
        img_array = np.array(img)
        red_mean = float(np.mean(img_array[:,:,0]))
        red_std = float(np.std(img_array[:,:,0]))
        green_mean = float(np.mean(img_array[:,:,1]))
        green_std = float(np.std(img_array[:,:,1]))
        blue_mean = float(np.mean(img_array[:,:,2]))
        blue_std = float(np.std(img_array[:,:,2]))
        
        return {
            'path': img_path,
            'label': label,
            'ela_std': ela_std,
            'ela_mean': ela_mean,
            'ela_max': ela_max,
            'log_hf_ratio': log_hf_ratio,
            'tail_70': tail_70,
            'tail_80': tail_80,
            'tail_90': tail_90,
            'wavelet_cH_std': wavelet_cH_std,
            'wavelet_cV_std': wavelet_cV_std,
            'wavelet_cD_std': wavelet_cD_std,
            'wavelet_energy': wavelet_energy,
            'lbp_entropy': lbp_entropy,
            'lbp_std': lbp_std,
            'gradient_mean': gradient_mean,
            'gradient_std': gradient_std,
            'gradient_max': gradient_max,
            'gradient_skew': gradient_skew,
            'red_mean': red_mean,
            'red_std': red_std,
            'green_mean': green_mean,
            'green_std': green_std,
            'blue_mean': blue_mean,
            'blue_std': blue_std
        }
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None

# Main training
if __name__ == "__main__":
    print("="*70)
    print("RETRAINING MODEL - KAGGLE ART DATASET")
    print("="*70)
    
    # Dataset paths
    BASE_PATH = "C:/Users/Vatase Radu/Downloads/DATASET_antrenare"
    
    print(f"\n📂 Dataset: {BASE_PATH}")
    print("   Cautare recursiva imagini...")
    
    # Find all images recursively
    real_files = glob.glob(os.path.join(BASE_PATH, "RealArt", "**", "*.*"), recursive=True)
    fake_files = glob.glob(os.path.join(BASE_PATH, "AiArtData", "**", "*.*"), recursive=True)
    
    # Filter only images
    valid_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    real_files = [f for f in real_files if os.path.splitext(f)[1] in valid_extensions]
    fake_files = [f for f in fake_files if os.path.splitext(f)[1] in valid_extensions]
    
    print(f"   REAL: {len(real_files)} imagini")
    print(f"   FAKE: {len(fake_files)} imagini")
    print(f"   Total: {len(real_files) + len(fake_files)} imagini")
    
    # Prepare tasks
    tasks = [(f, 0) for f in real_files] + [(f, 1) for f in fake_files]
    
    # Extract features
    print(f"\n🚀 Procesare imagini...")
    results = []
    for img_path, label in tqdm(tasks, desc="Extragere features"):
        result = extract_features_from_path(img_path, label)
        if result is not None:
            results.append(result)
    
    print(f"✅ Procesate: {len(results)}/{len(tasks)} imagini ({len(results)/len(tasks)*100:.1f}%)")
    
    if len(results) < 100:
        print("❌ Prea puține imagini procesate cu succes!")
        exit(1)
    
    # Prepare data
    feature_names = ['ela_std', 'ela_mean', 'ela_max', 'log_hf_ratio', 'tail_70', 
                     'tail_80', 'tail_90', 'wavelet_cH_std', 'wavelet_cV_std', 
                     'wavelet_cD_std', 'wavelet_energy', 'lbp_entropy', 'lbp_std',
                     'gradient_mean', 'gradient_std', 'gradient_max', 'gradient_skew',
                     'red_mean', 'red_std', 'green_mean', 'green_std', 'blue_mean', 'blue_std']
    
    X = np.array([[r[f] for f in feature_names] for r in results])
    y = np.array([r['label'] for r in results])
    
    print(f"\n📊 Dataset final:")
    print(f"   Features: {X.shape[1]} (23 features forensice)")
    print(f"   Samples: {X.shape[0]}")
    print(f"   REAL (0): {np.sum(y==0)}")
    print(f"   FAKE (1): {np.sum(y==1)}")
    print(f"   Balance: {np.sum(y==0)/len(y)*100:.1f}% REAL, {np.sum(y==1)/len(y)*100:.1f}% FAKE")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    print(f"\n🔄 Split:")
    print(f"   Training: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"   Testing: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest
    print(f"\n🌲 Training Random Forest Classifier...")
    print(f"   Hyperparameters:")
    print(f"   - n_estimators: 200")
    print(f"   - max_depth: 20")
    print(f"   - class_weight: balanced")
    
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    print(f"\n📈 Evaluare Model:")
    
    train_pred = model.predict(X_train_scaled)
    train_acc = accuracy_score(y_train, train_pred)
    print(f"   Training Accuracy: {train_acc*100:.2f}%")
    
    test_pred = model.predict(X_test_scaled)
    test_acc = accuracy_score(y_test, test_pred)
    print(f"   Test Accuracy: {test_acc*100:.2f}%")
    
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
    print(f"   5-Fold CV: {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, test_pred)
    print(f"\n   Confusion Matrix:")
    print(f"                Predicted")
    print(f"              REAL  FAKE")
    print(f"   Actual REAL  {cm[0,0]:4d}  {cm[0,1]:4d}")
    print(f"         FAKE  {cm[1,0]:4d}  {cm[1,1]:4d}")
    
    tn, fp, fn, tp = cm.ravel()
    precision_real = tn / (tn + fn) if (tn + fn) > 0 else 0
    recall_real = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision_fake = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_fake = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    print(f"\n   Per-Class Metrics:")
    print(f"   REAL: Precision={precision_real:.2%}, Recall={recall_real:.2%}")
    print(f"   FAKE: Precision={precision_fake:.2%}, Recall={recall_fake:.2%}")
    
    # Feature Importances
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    print(f"\n🔑 Top 10 Most Important Features:")
    for i in range(min(10, len(feature_names))):
        idx = indices[i]
        print(f"   {i+1}. {feature_names[idx]}: {importances[idx]:.4f}")
    
    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_data = {
        'model': model,
        'scaler': scaler,
        'feature_names': feature_names,
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'cv_scores': cv_scores,
        'confusion_matrix': cm,
        'timestamp': timestamp,
        'dataset': 'KaggleArt',
        'n_samples': len(results)
    }
    
    with open('final_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    print(f"\n💾 Model salvat ca: final_model.pkl")
    
    backup_name = f'model_kaggle_art_{timestamp}_acc{test_acc*100:.1f}.pkl'
    with open(backup_name, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"💾 Backup salvat ca: {backup_name}")
    
    # Plot feature importances
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(importances)), importances[indices])
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title(f'Feature Importances - Test Accuracy: {test_acc*100:.1f}%')
    plt.tight_layout()
    plt.savefig(f'feature_importances_kaggle_{timestamp}.png', dpi=150)
    print(f"📊 Grafic salvat ca: feature_importances_kaggle_{timestamp}.png")
    
    # Summary
    print("\n" + "="*70)
    print("REZUMAT FINAL")
    print("="*70)
    print(f"✅ Model antrenat cu succes!")
    print(f"   Dataset: {len(results)} imagini (Kaggle Art/Photos)")
    print(f"   Test Accuracy: {test_acc*100:.2f}%")
    print(f"   Cross-Validation: {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")
    
    if test_acc >= 0.80:
        print(f"\n🎉 EXCELLENT! Accuracy > 80% - modelul e gata pentru producție!")
    elif test_acc >= 0.70:
        print(f"\n✅ BINE! Accuracy > 70% - modelul e funcțional cu reguli forensice")
    else:
        print(f"\n⚠️ MEDIU - Accuracy < 70% - consideră mai multe date sau alt dataset")
    
    print(f"\n🎯 Următorii pași:")
    print(f"   1. Repornește aplicația Streamlit")
    print(f"   2. Testează cu imagini diverse")
    print(f"   3. Verifică dacă predicțiile sunt corecte")
    print("="*70)
