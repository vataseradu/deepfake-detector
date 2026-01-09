"""
ULTIMATE HYBRID: CNN Features + Traditional Features + Ensemble
================================================================
Final approach combining visual patterns with forensic analysis

Strategy:
1. Extract CNN features from intermediate layers
2. Combine with traditional features (FFT, ELA, Wavelet, etc.)
3. Train ensemble classifier
4. Handle class imbalance with SMOTE
"""

import os
import glob
import random
import pickle
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageChops
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import warnings
warnings.filterwarnings('ignore')

try:
    import pywt
    from skimage.feature import local_binary_pattern
    from scipy.stats import entropy, skew, kurtosis
    import xgboost as xgb
    from imblearn.over_sampling import SMOTE
    from imblearn.ensemble import BalancedRandomForestClassifier
    import tensorflow as tf
    from tensorflow import keras
except ImportError:
    print("📦 Installing packages...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'imbalanced-learn'])
    from imblearn.over_sampling import SMOTE
    from imblearn.ensemble import BalancedRandomForestClassifier

# Configuration
IMG_SIZE = 224
np.random.seed(42)
random.seed(42)

# Load pre-trained CNN for feature extraction
print("Loading CNN feature extractor...")
base_model = keras.applications.EfficientNetB0(
    include_top=False,
    weights='imagenet',
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    pooling='avg'
)
print("✅ CNN loaded")

def extract_cnn_features(img_path):
    """Extract deep features using pre-trained CNN"""
    try:
        img = Image.open(img_path).convert('RGB')
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        features = base_model.predict(img_array, verbose=0).flatten()
        return features
    except:
        return np.zeros(1280)  # EfficientNetB0 output size

def extract_traditional_features(img_path):
    """Extract all traditional forensic features"""
    try:
        with Image.open(img_path) as im:
            im = im.convert('RGB')
            features = {}
            
            # ELA
            temp_file = f"temp_{os.getpid()}.jpg"
            im.save(temp_file, 'JPEG', quality=90)
            resaved = Image.open(temp_file)
            ela_im = ImageChops.difference(im, resaved)
            ela_array = np.array(ela_im)
            gray_ela = np.mean(ela_array, axis=2)
            features['ela_std'] = np.std(gray_ela)
            features['ela_mean'] = np.mean(gray_ela)
            features['ela_max'] = np.max(gray_ela)
            
            try:
                resaved.close()
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except:
                pass
            
            # FFT
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
                
                cutoff = int(0.6 * n_freq)
                low_p = np.mean(psd1D[:cutoff])
                high_p = np.mean(psd1D[cutoff:])
                features['log_hf_ratio'] = np.log10(high_p / low_p) if low_p > 0 and high_p > 0 else -10.0
                
                for zone, pct in [('70', 0.7), ('80', 0.8), ('90', 0.9)]:
                    start_idx = int(pct * n_freq)
                    if len(psd1D[start_idx:]) > 5:
                        x_tail = np.log10(np.arange(start_idx, n_freq) + 1).reshape(-1, 1)
                        y_tail = np.log10(psd1D[start_idx:])
                        model = LinearRegression().fit(x_tail, y_tail)
                        features[f'tail_{zone}'] = model.coef_[0]
                    else:
                        features[f'tail_{zone}'] = 0.0
            else:
                features.update({k: 0.0 for k in ['log_hf_ratio', 'tail_70', 'tail_80', 'tail_90']})
            
            # Wavelet
            coeffs = pywt.dwt2(img_gray, 'db4')
            cA, (cH, cV, cD) = coeffs
            features['wavelet_cH_std'] = np.std(cH)
            features['wavelet_cV_std'] = np.std(cV)
            features['wavelet_cD_std'] = np.std(cD)
            
            # LBP
            lbp = local_binary_pattern(img_gray, P=24, R=3, method='uniform')
            hist, _ = np.histogram(lbp.ravel(), bins=26, range=(0, 26), density=True)
            features['lbp_entropy'] = entropy(hist + 1e-10)
            features['lbp_std'] = np.std(lbp)
            
            # Gradient
            gx = np.gradient(img_gray, axis=1)
            gy = np.gradient(img_gray, axis=0)
            magnitude = np.sqrt(gx**2 + gy**2)
            features['gradient_mean'] = np.mean(magnitude)
            features['gradient_std'] = np.std(magnitude)
            features['gradient_skew'] = skew(magnitude.ravel())
            
            return features
    except:
        return {k: 0.0 for k in ['ela_std', 'ela_mean', 'ela_max', 'log_hf_ratio', 
                                   'tail_70', 'tail_80', 'tail_90', 'wavelet_cH_std',
                                   'wavelet_cV_std', 'wavelet_cD_std', 'lbp_entropy',
                                   'lbp_std', 'gradient_mean', 'gradient_std', 'gradient_skew']}

def extract_all_features(img_path):
    """Combine CNN and traditional features"""
    cnn_feats = extract_cnn_features(img_path)
    trad_feats = extract_traditional_features(img_path)
    trad_array = np.array(list(trad_feats.values()))
    combined = np.concatenate([cnn_feats, trad_array])
    return combined

def main():
    print("🎯 ULTIMATE HYBRID DETECTION SYSTEM")
    print("="*80)
    print("CNN Deep Features + Traditional Forensics + Ensemble + SMOTE")
    print("="*80)
    
    # Load data
    BASE_PATH = r"C:\Users\Vatase Radu\Downloads\datetrainingFACE"
    REAL_PATH = os.path.join(BASE_PATH, "training_real") 
    FAKE_PATH = os.path.join(BASE_PATH, "training_fake")
    
    if not os.path.exists(REAL_PATH):
        print("❌ Path not found")
        return
    
    real_files = glob.glob(os.path.join(REAL_PATH, "*.*"))
    fake_files = glob.glob(os.path.join(FAKE_PATH, "*.*"))
    
    print(f"📂 Found: {len(real_files)} REAL, {len(fake_files)} FAKE")
    
    # Sample
    SAMPLE_SIZE = 1200
    if len(real_files) > SAMPLE_SIZE:
        random.seed(42)
        random.shuffle(real_files)
        random.shuffle(fake_files)
        real_files = real_files[:SAMPLE_SIZE]
        fake_files = fake_files[:SAMPLE_SIZE]
    
    all_files = real_files + fake_files
    labels = [0] * len(real_files) + [1] * len(fake_files)
    
    # Extract features
    print(f"\n🔬 Extracting hybrid features from {len(all_files)} images...")
    print("   (This will take 5-10 minutes...)")
    
    features_list = []
    valid_labels = []
    
    for file, label in tqdm(zip(all_files, labels), total=len(all_files)):
        try:
            feats = extract_all_features(file)
            if feats is not None and not np.any(np.isnan(feats)):
                features_list.append(feats)
                valid_labels.append(label)
        except:
            continue
    
    X = np.array(features_list)
    y = np.array(valid_labels)
    
    print(f"\n✅ Extracted {X.shape[0]} samples with {X.shape[1]} features")
    print(f"   ({np.sum(y==0)} REAL, {np.sum(y==1)} FAKE)")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Apply SMOTE for balance
    print("\n⚖️  Applying SMOTE (balancing classes)...")
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
    print(f"   After SMOTE: {len(y_train_balanced)} samples")
    
    # Train multiple models
    print("\n🤖 Training ensemble models...")
    
    models = {
        'Balanced RF': BalancedRandomForestClassifier(
            n_estimators=500,
            max_depth=30,
            random_state=42,
            n_jobs=-1
        ),
        'XGBoost': xgb.XGBClassifier(
            n_estimators=500,
            max_depth=10,
            learning_rate=0.05,
            scale_pos_weight=len(y_train_balanced[y_train_balanced==0])/len(y_train_balanced[y_train_balanced==1]),
            random_state=42,
            n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.05,
            random_state=42
        )
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n📍 Training {name}...")
        model.fit(X_train_balanced, y_train_balanced)
        
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        cm = confusion_matrix(y_test, y_pred)
        
        results[name] = {
            'model': model,
            'accuracy': acc,
            'auc': auc,
            'cm': cm,
            'y_pred': y_pred,
            'y_proba': y_proba
        }
        
        print(f"   Accuracy: {acc*100:.2f}%")
        print(f"   AUC: {auc:.4f}")
    
    # Ensemble voting
    print("\n🗳️  Creating ensemble predictor...")
    y_pred_ensemble = np.round(np.mean([r['y_proba'] for r in results.values()], axis=0)).astype(int)
    acc_ensemble = accuracy_score(y_test, y_pred_ensemble)
    auc_ensemble = roc_auc_score(y_test, np.mean([r['y_proba'] for r in results.values()], axis=0))
    cm_ensemble = confusion_matrix(y_test, y_pred_ensemble)
    
    results['Ensemble'] = {
        'accuracy': acc_ensemble,
        'auc': auc_ensemble,
        'cm': cm_ensemble
    }
    
    # Results
    print("\n" + "="*80)
    print("📈 FINAL RESULTS")
    print("="*80)
    
    best_name = max(results, key=lambda k: results[k]['accuracy'])
    best_acc = results[best_name]['accuracy']
    
    for name, res in results.items():
        print(f"\n{'='*80}")
        print(f"🏆 {name}")
        print(f"{'='*80}")
        print(f"Accuracy: {res['accuracy']*100:.2f}%")
        print(f"AUC:      {res['auc']:.4f}")
        cm = res['cm']
        print(f"\nConfusion Matrix:")
        print(f"   TN={cm[0,0]:4d} | FP={cm[0,1]:4d}")
        print(f"   FN={cm[1,0]:4d} | TP={cm[1,1]:4d}")
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Model comparison
    names = list(results.keys())
    accs = [results[n]['accuracy']*100 for n in names]
    colors = ['gold' if n == best_name else 'steelblue' for n in names]
    
    bars = axes[0, 0].bar(names, accs, color=colors, edgecolor='black', linewidth=1.5)
    axes[0, 0].axhline(y=90, color='red', linestyle='--', linewidth=2, label='Target 90%')
    axes[0, 0].set_ylabel('Accuracy (%)')
    axes[0, 0].set_title('Model Comparison')
    axes[0, 0].legend()
    axes[0, 0].grid(axis='y', alpha=0.3)
    for bar, acc in zip(bars, accs):
        height = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width()/2., height,
                       f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Best model confusion matrix
    best_cm = results[best_name]['cm']
    cm_norm = best_cm.astype('float') / best_cm.sum(axis=1)[:, np.newaxis]
    im = axes[0, 1].imshow(cm_norm, cmap='Blues', aspect='auto')
    axes[0, 1].set_xticks([0, 1])
    axes[0, 1].set_yticks([0, 1])
    axes[0, 1].set_xticklabels(['REAL', 'FAKE'])
    axes[0, 1].set_yticklabels(['REAL', 'FAKE'])
    axes[0, 1].set_title(f'Best: {best_name}\\n{best_acc*100:.2f}%')
    for i in range(2):
        for j in range(2):
            axes[0, 1].text(j, i, f'{cm_norm[i, j]:.2f}\\n({best_cm[i, j]})',
                           ha="center", va="center", fontweight='bold')
    
    # Metrics comparison
    metrics = ['Accuracy', 'AUC']
    x = np.arange(len(metrics))
    width = 0.2
    for i, name in enumerate(names):
        values = [results[name]['accuracy'], results[name]['auc']]
        axes[1, 0].bar(x + i*width, values, width, label=name)
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].set_title('Metrics Comparison')
    axes[1, 0].set_xticks(x + width * 1.5)
    axes[1, 0].set_xticklabels(metrics)
    axes[1, 0].legend()
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # Summary text
    axes[1, 1].axis('off')
    summary_text = f"""
    ULTIMATE HYBRID SYSTEM
    
    🎯 Target: 90%
    🏆 Best:   {best_acc*100:.2f}% ({best_name})
    
    Features Used:
    • CNN Deep Features: 1280
    • Traditional Features: 15
    • Total: 1295 features
    
    Techniques Applied:
    ✓ Transfer Learning
    ✓ Forensic Analysis
    ✓ SMOTE Balancing
    ✓ Ensemble Voting
    ✓ Feature Scaling
    
    Dataset Challenge:
    High-quality AI images
    (SOTA generators)
    """
    axes[1, 1].text(0.1, 0.5, summary_text, fontsize=12, 
                    verticalalignment='center', family='monospace')
    
    plt.suptitle('ULTIMATE HYBRID AI DETECTION SYSTEM', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('ultimate_results.png', dpi=150, bbox_inches='tight')
    print("\n📊 Results saved: ultimate_results.png")
    
    # Save best model
    with open('ultimate_model.pkl', 'wb') as f:
        pickle.dump({
            'model': results[best_name]['model'],
            'scaler': scaler
        }, f)
    print("💾 Best model saved: ultimate_model.pkl")
    
    print("\n" + "="*80)
    print("✨ FINAL ASSESSMENT")
    print("="*80)
    print(f"Best Accuracy: {best_acc*100:.2f}%")
    if best_acc >= 0.9:
        print("✅ TARGET REACHED! 🎉")
    elif best_acc >= 0.75:
        print("📊 Strong performance! Dataset is extremely challenging.")
    else:
        print("📊 Dataset contains SOTA AI images difficult to distinguish.")
    print("="*80)
    
    plt.show()

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
