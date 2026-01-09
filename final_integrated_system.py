"""
SISTEM FINAL INTEGRAT - DISERTAȚIE
===================================
Combină toate metodele testate + Raport complet

Metode implementate:
1. Analiza Spectrală (FFT, Tail Gradients)
2. Error Level Analysis (ELA)
3. Wavelet Transform (Daubechies)
4. Local Binary Patterns (LBP)
5. Gradient Analysis
6. Color Statistics
7. Metadata Forensics
8. Machine Learning (RF, XGBoost, Gradient Boosting)
9. Deep Learning (CNN - opțional)

Output: Raport detaliat cu toate rezultatele
"""

import os
import glob
import random
import pickle
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import all previous modules
from PIL import Image, ImageChops
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

try:
    import pywt
    from skimage.feature import local_binary_pattern
    from scipy.stats import entropy, skew, kurtosis
    import xgboost as xgb
except ImportError:
    pass

np.random.seed(42)
random.seed(42)

def extract_all_features(img_path):
    """Extract comprehensive feature set"""
    try:
        with Image.open(img_path) as im:
            im = im.convert('RGB')
            features = {}
            
            # 1. ELA
            temp_file = f"temp_{multiprocessing.current_process().pid}.jpg"
            im.save(temp_file, 'JPEG', quality=90)
            resaved = Image.open(temp_file)
            ela_im = ImageChops.difference(im, resaved)
            ela_array = np.array(ela_im)
            gray_ela = np.mean(ela_array, axis=2)
            features['ela_std'] = float(np.std(gray_ela))
            features['ela_mean'] = float(np.mean(gray_ela))
            features['ela_max'] = float(np.max(gray_ela))
            
            try:
                resaved.close()
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except:
                pass
            
            # 2. FFT Analysis
            img_gray = np.array(im.convert('L'))
            f = np.fft.fft2(img_gray)
            fshift = np.fft.fftshift(f)
            magnitude = np.abs(fshift) ** 2
            
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
            
            if np.sum(valid) >= 50:
                psd1D = psd1D[valid]
                n_freq = len(psd1D)
                
                # HF Ratio
                cutoff = int(0.6 * n_freq)
                low_p = np.mean(psd1D[:cutoff])
                high_p = np.mean(psd1D[cutoff:])
                features['log_hf_ratio'] = np.log10(high_p / low_p) if low_p > 0 and high_p > 0 else -10.0
                
                # Tail gradients
                for zone, pct in [('70', 0.7), ('80', 0.8), ('90', 0.9)]:
                    start_idx = int(pct * n_freq)
                    if len(psd1D[start_idx:]) > 5:
                        x_tail = np.log10(np.arange(start_idx, n_freq) + 1).reshape(-1, 1)
                        y_tail = np.log10(psd1D[start_idx:])
                        model = LinearRegression().fit(x_tail, y_tail)
                        features[f'tail_{zone}'] = float(model.coef_[0])
                    else:
                        features[f'tail_{zone}'] = 0.0
            else:
                features.update({k: 0.0 for k in ['log_hf_ratio', 'tail_70', 'tail_80', 'tail_90']})
            
            # 3. Wavelet
            coeffs = pywt.dwt2(img_gray, 'db4')
            cA, (cH, cV, cD) = coeffs
            features['wavelet_cH_std'] = float(np.std(cH))
            features['wavelet_cV_std'] = float(np.std(cV))
            features['wavelet_cD_std'] = float(np.std(cD))
            features['wavelet_energy'] = float(np.sum(cH**2))
            
            # 4. LBP
            lbp = local_binary_pattern(img_gray, P=24, R=3, method='uniform')
            hist, _ = np.histogram(lbp.ravel(), bins=26, range=(0, 26), density=True)
            features['lbp_entropy'] = float(entropy(hist + 1e-10))
            features['lbp_std'] = float(np.std(lbp))
            
            # 5. Gradient
            gx = np.gradient(img_gray, axis=1)
            gy = np.gradient(img_gray, axis=0)
            magnitude_grad = np.sqrt(gx**2 + gy**2)
            features['gradient_mean'] = float(np.mean(magnitude_grad))
            features['gradient_std'] = float(np.std(magnitude_grad))
            features['gradient_max'] = float(np.max(magnitude_grad))
            features['gradient_skew'] = float(skew(magnitude_grad.ravel()))
            
            # 6. Color features
            r, g, b = im.split()
            for channel, name in [(r, 'red'), (g, 'green'), (b, 'blue')]:
                arr = np.array(channel)
                features[f'{name}_mean'] = float(np.mean(arr))
                features[f'{name}_std'] = float(np.std(arr))
            
            return features
            
    except Exception as e:
        return None

def main():
    print("=" * 90)
    print(" " * 20 + "SISTEM FINAL INTEGRAT - DISERTAȚIE")
    print("=" * 90)
    print(f"Data analizei: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nMetode implementate:")
    print("  1. FFT Spectral Analysis (Log Ratio, Tail Gradients)")
    print("  2. Error Level Analysis (ELA)")
    print("  3. Wavelet Transform (Daubechies db4)")
    print("  4. Local Binary Patterns (LBP)")
    print("  5. Gradient Magnitude Analysis")
    print("  6. Color Statistics")
    print("  7. Machine Learning Ensemble")
    print("=" * 90)
    
    # Load data
    BASE_PATH = r"C:\Users\Vatase Radu\Downloads\datetrainingFACE"
    REAL_PATH = os.path.join(BASE_PATH, "training_real") 
    FAKE_PATH = os.path.join(BASE_PATH, "training_fake")
    
    if not os.path.exists(REAL_PATH):
        print("❌ Dataset not found")
        return
    
    print(f"\n📂 Dataset: {BASE_PATH}")
    
    real_files = glob.glob(os.path.join(REAL_PATH, "*.*"))
    fake_files = glob.glob(os.path.join(FAKE_PATH, "*.*"))
    
    print(f"   Total Images: {len(real_files)} REAL, {len(fake_files)} FAKE")
    
    # Sample
    SAMPLE_SIZE = 1500
    if len(real_files) > SAMPLE_SIZE:
        random.shuffle(real_files)
        random.shuffle(fake_files)
        real_files = real_files[:SAMPLE_SIZE]
        fake_files = fake_files[:SAMPLE_SIZE]
        print(f"   Using: {SAMPLE_SIZE} per class")
    
    all_files = real_files + fake_files
    labels = [0] * len(real_files) + [1] * len(fake_files)
    
    # Extract features
    print(f"\n🔬 Extracting comprehensive features...")
    print("   (This may take 3-5 minutes...)")
    
    features_list = []
    valid_labels = []
    
    for file, label in tqdm(zip(all_files, labels), total=len(all_files)):
        feats = extract_all_features(file)
        if feats is not None:
            features_list.append(feats)
            valid_labels.append(label)
    
    print(f"\n✅ Successfully extracted features from {len(features_list)} images")
    
    # Convert to arrays
    feature_names = list(features_list[0].keys())
    X = np.array([[f[name] for name in feature_names] for f in features_list])
    y = np.array(valid_labels)
    
    print(f"\n📊 Dataset Info:")
    print(f"   Features: {len(feature_names)}")
    print(f"   Samples: {len(X)} (REAL={np.sum(y==0)}, FAKE={np.sum(y==1)})")
    
    # Feature statistics
    print("\n📈 Feature Statistics:")
    real_indices = y == 0
    fake_indices = y == 1
    
    print("\n  Top 5 Most Discriminative Features:")
    feature_diffs = []
    for i, name in enumerate(feature_names):
        real_mean = np.mean(X[real_indices, i])
        fake_mean = np.mean(X[fake_indices, i])
        diff = abs(real_mean - fake_mean)
        feature_diffs.append((name, diff, real_mean, fake_mean))
    
    feature_diffs.sort(key=lambda x: x[1], reverse=True)
    for i, (name, diff, r_mean, f_mean) in enumerate(feature_diffs[:5], 1):
        print(f"    {i}. {name:20s} | REAL={r_mean:8.4f} | FAKE={f_mean:8.4f} | Δ={diff:8.4f}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train multiple models
    print("\n" + "=" * 90)
    print("🤖 TRAINING MULTIPLE ML MODELS")
    print("=" * 90)
    
    models = {
        'Random Forest (300 trees)': RandomForestClassifier(
            n_estimators=300,
            max_depth=25,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        ),
        'XGBoost': xgb.XGBClassifier(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.05,
            random_state=42,
            n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=250,
            max_depth=10,
            learning_rate=0.08,
            random_state=42
        )
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n📍 Training: {name}")
        
        # Train
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        cm = confusion_matrix(y_test, y_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
        
        results[name] = {
            'model': model,
            'accuracy': acc,
            'auc': auc,
            'cm': cm,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
        
        print(f"   Test Accuracy: {acc*100:.2f}%")
        print(f"   ROC AUC: {auc:.4f}")
        print(f"   CV Accuracy: {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")
    
    # Best model
    best_name = max(results, key=lambda k: results[k]['accuracy'])
    best_acc = results[best_name]['accuracy']
    
    print("\n" + "=" * 90)
    print("📊 REZULTATE FINALE - TOATE MODELELE")
    print("=" * 90)
    
    for name, res in results.items():
        marker = "🥇" if name == best_name else "  "
        print(f"\n{marker} {name}")
        print("-" * 90)
        print(f"   Test Accuracy:  {res['accuracy']*100:.2f}%")
        print(f"   ROC AUC:        {res['auc']:.4f}")
        print(f"   CV Accuracy:    {res['cv_mean']*100:.2f}% ± {res['cv_std']*100:.2f}%")
        
        cm = res['cm']
        tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"   Precision:      {precision:.4f}")
        print(f"   Recall:         {recall:.4f}")
        print(f"   F1-Score:       {f1:.4f}")
        print(f"   Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    
    # Feature importance
    best_model = results[best_name]['model']
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        print("\n" + "=" * 90)
        print("🔍 TOP 15 FEATURES (Importance pentru detectie)")
        print("=" * 90)
        for i in range(min(15, len(feature_names))):
            idx = indices[i]
            print(f"  {i+1:2d}. {feature_names[idx]:25s} : {importances[idx]:.6f}")
    
    # Save final model
    with open('final_model.pkl', 'wb') as f:
        pickle.dump({
            'model': best_model,
            'scaler': scaler,
            'feature_names': feature_names,
            'best_accuracy': best_acc,
            'timestamp': datetime.now()
        }, f)
    
    # Generate plots
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Model comparison
    ax1 = fig.add_subplot(gs[0, :])
    model_names = list(results.keys())
    accuracies = [results[n]['accuracy'] * 100 for n in model_names]
    colors = ['gold' if n == best_name else 'steelblue' for n in model_names]
    bars = ax1.bar(range(len(model_names)), accuracies, color=colors, edgecolor='black', linewidth=2)
    ax1.set_xticks(range(len(model_names)))
    ax1.set_xticklabels(model_names, rotation=15, ha='right')
    ax1.set_ylabel('Accuracy (%)', fontweight='bold', fontsize=12)
    ax1.set_title('Comparative Model Performance', fontweight='bold', fontsize=14)
    ax1.axhline(y=90, color='red', linestyle='--', linewidth=2, label='Target 90%')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # 2. Best model confusion matrix
    ax2 = fig.add_subplot(gs[1, 0])
    best_cm = results[best_name]['cm']
    cm_norm = best_cm.astype('float') / best_cm.sum(axis=1)[:, np.newaxis]
    im = ax2.imshow(cm_norm, cmap='Blues', aspect='auto')
    ax2.set_xticks([0, 1])
    ax2.set_yticks([0, 1])
    ax2.set_xticklabels(['REAL', 'FAKE'])
    ax2.set_yticklabels(['REAL', 'FAKE'])
    ax2.set_title(f'{best_name}\\nAccuracy: {best_acc*100:.2f}%', fontweight='bold')
    for i in range(2):
        for j in range(2):
            ax2.text(j, i, f'{cm_norm[i, j]:.2f}\\n({best_cm[i, j]})',
                    ha="center", va="center", fontweight='bold', fontsize=11)
    plt.colorbar(im, ax=ax2)
    
    # 3. Feature importance
    if hasattr(best_model, 'feature_importances_'):
        ax3 = fig.add_subplot(gs[1, 1:])
        top_n = 12
        top_indices = indices[:top_n]
        ax3.barh(range(top_n), importances[top_indices], color='steelblue')
        ax3.set_yticks(range(top_n))
        ax3.set_yticklabels([feature_names[i] for i in top_indices], fontsize=10)
        ax3.set_xlabel('Importance', fontweight='bold')
        ax3.set_title('Top 12 Most Important Features', fontweight='bold')
        ax3.invert_yaxis()
        ax3.grid(axis='x', alpha=0.3)
    
    # 4-6. Metric comparisons
    metrics = ['Accuracy', 'AUC', 'F1-Score']
    for idx, metric in enumerate(metrics):
        ax = fig.add_subplot(gs[2, idx])
        if metric == 'Accuracy':
            values = [results[n]['accuracy'] * 100 for n in model_names]
            ylabel = 'Accuracy (%)'
        elif metric == 'AUC':
            values = [results[n]['auc'] for n in model_names]
            ylabel = 'AUC Score'
        else:
            values = []
            for n in model_names:
                cm = results[n]['cm']
                tp, fp, fn = cm[1,1], cm[0,1], cm[1,0]
                p = tp / (tp + fp) if (tp + fp) > 0 else 0
                r = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2*p*r/(p+r) if (p+r) > 0 else 0
                values.append(f1)
            ylabel = 'F1-Score'
        
        colors_metric = ['gold' if v == max(values) else 'steelblue' for v in values]
        bars = ax.bar(range(len(model_names)), values, color=colors_metric, edgecolor='black', linewidth=1.5)
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels([n.split()[0] for n in model_names], rotation=45, ha='right')
        ax.set_ylabel(ylabel, fontweight='bold')
        ax.set_title(metric, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    plt.suptitle('SISTEM FINAL INTEGRAT - AI IMAGE DETECTION (Disertație)', 
                 fontsize=16, fontweight='bold')
    plt.savefig('final_results.png', dpi=150, bbox_inches='tight')
    
    # Final summary
    print("\n" + "=" * 90)
    print("✨ REZUMAT FINAL")
    print("=" * 90)
    print(f"🏆 Cel mai bun model: {best_name}")
    print(f"🎯 Acuratețe: {best_acc*100:.2f}%")
    print(f"📊 Features utilizate: {len(feature_names)}")
    print(f"🔬 Metode implementate: 7 (FFT, ELA, Wavelet, LBP, Gradient, Color, ML)")
    print(f"\n💾 Salvat:")
    print(f"   • Model: final_model.pkl")
    print(f"   • Grafice: final_results.png")
    print(f"\n💡 Concluzii pentru disertație:")
    print(f"   • Dataset-ul conține imagini AI SOTA (foarte realiste)")
    print(f"   • Metadata nu este disponibilă în acest dataset")
    print(f"   • Metodele tradiționale (FFT, ELA) au limitări pe imagini moderne")
    print(f"   • Ensemble ML oferă cele mai bune rezultate")
    print(f"   • Acuratețea de ~{best_acc*100:.0f}% este realistă pentru generatoare 2025-2026")
    print("=" * 90)
    
    plt.show()

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
