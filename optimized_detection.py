"""
ULTRA-OPTIMIZED AI IMAGE DETECTION SYSTEM
==========================================
Target: 90%+ Accuracy through Advanced ML Techniques

Strategies:
1. Enhanced Feature Engineering (30+ features)
2. Multiple Models (RF, XGBoost, SVM, Neural Network)
3. Hyperparameter Optimization (Grid Search)
4. Ensemble Voting (Stacking)
5. Feature Selection & Scaling
"""

import os
import glob
import csv
import random
import pickle
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageChops, ImageFilter, ImageStat
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif
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
except ImportError:
    print("📦 Installing missing packages...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'PyWavelets', 'scikit-image', 'xgboost'])
    import pywt
    from skimage.feature import local_binary_pattern
    from scipy.stats import entropy, skew, kurtosis
    import xgboost as xgb

# === ENHANCED FEATURE EXTRACTION ===

def extract_color_features(im):
    """Features din distribuția culorilor"""
    try:
        # Color channel statistics
        r, g, b = im.split()
        features = {}
        
        for channel, name in [(r, 'red'), (g, 'green'), (b, 'blue')]:
            arr = np.array(channel)
            features[f'{name}_mean'] = float(np.mean(arr))
            features[f'{name}_std'] = float(np.std(arr))
            features[f'{name}_skew'] = float(skew(arr.ravel()))
            features[f'{name}_kurtosis'] = float(kurtosis(arr.ravel()))
        
        # Color correlations
        r_arr = np.array(r).ravel()
        g_arr = np.array(g).ravel()
        b_arr = np.array(b).ravel()
        features['rg_corr'] = float(np.corrcoef(r_arr, g_arr)[0, 1])
        features['rb_corr'] = float(np.corrcoef(r_arr, b_arr)[0, 1])
        features['gb_corr'] = float(np.corrcoef(g_arr, b_arr)[0, 1])
        
        return features
    except:
        return {f'{c}_{s}': 0.0 for c in ['red', 'green', 'blue'] for s in ['mean', 'std', 'skew', 'kurtosis']} | {'rg_corr': 0.0, 'rb_corr': 0.0, 'gb_corr': 0.0}

def extract_noise_features(img_array):
    """Noise pattern analysis"""
    try:
        # High-pass filter for noise
        from scipy import ndimage
        high_pass = img_array - ndimage.gaussian_filter(img_array, sigma=2)
        
        features = {
            'noise_std': float(np.std(high_pass)),
            'noise_mean': float(np.mean(np.abs(high_pass))),
            'noise_max': float(np.max(np.abs(high_pass)))
        }
        return features
    except:
        return {'noise_std': 0.0, 'noise_mean': 0.0, 'noise_max': 0.0}

def extract_wavelet_features(img_array):
    """Enhanced wavelet analysis"""
    try:
        coeffs = pywt.dwt2(img_array, 'db4')
        cA, (cH, cV, cD) = coeffs
        
        features = {
            'wavelet_cH_std': float(np.std(cH)),
            'wavelet_cV_std': float(np.std(cV)),
            'wavelet_cD_std': float(np.std(cD)),
            'wavelet_cH_energy': float(np.sum(cH**2)),
            'wavelet_detail_ratio': float(np.sum(cH**2) / (np.sum(cA**2) + 1e-10)),
            'wavelet_cH_mean': float(np.mean(np.abs(cH))),
            'wavelet_cV_mean': float(np.mean(np.abs(cV)))
        }
        return features
    except:
        return {k: 0.0 for k in ['wavelet_cH_std', 'wavelet_cV_std', 'wavelet_cD_std', 'wavelet_cH_energy', 'wavelet_detail_ratio', 'wavelet_cH_mean', 'wavelet_cV_mean']}

def extract_lbp_features(img_array):
    """LBP texture analysis"""
    try:
        lbp = local_binary_pattern(img_array, P=24, R=3, method='uniform')
        hist, _ = np.histogram(lbp.ravel(), bins=26, range=(0, 26), density=True)
        
        features = {
            'lbp_entropy': float(entropy(hist + 1e-10)),
            'lbp_uniformity': float(np.sum(hist**2)),
            'lbp_mean': float(np.mean(lbp)),
            'lbp_std': float(np.std(lbp))
        }
        return features
    except:
        return {k: 0.0 for k in ['lbp_entropy', 'lbp_uniformity', 'lbp_mean', 'lbp_std']}

def extract_gradient_features(img_array):
    """Enhanced gradient analysis"""
    try:
        gx = np.gradient(img_array, axis=1)
        gy = np.gradient(img_array, axis=0)
        magnitude = np.sqrt(gx**2 + gy**2)
        
        # Edge density
        edge_threshold = np.percentile(magnitude, 90)
        edge_density = np.sum(magnitude > edge_threshold) / magnitude.size
        
        features = {
            'gradient_mean': float(np.mean(magnitude)),
            'gradient_std': float(np.std(magnitude)),
            'gradient_max': float(np.max(magnitude)),
            'gradient_skew': float(skew(magnitude.ravel())),
            'gradient_kurtosis': float(kurtosis(magnitude.ravel())),
            'edge_density': float(edge_density)
        }
        return features
    except:
        return {k: 0.0 for k in ['gradient_mean', 'gradient_std', 'gradient_max', 'gradient_skew', 'gradient_kurtosis', 'edge_density']}

def analyze_image(file_info):
    """Extract ALL features from image"""
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
            temp_filename = f"temp_opt_{pid}.jpg"
            
            im.save(temp_filename, 'JPEG', quality=90)
            resaved = Image.open(temp_filename)
            ela_im = ImageChops.difference(im, resaved)
            
            ela_array = np.array(ela_im)
            gray_ela = np.mean(ela_array, axis=2)
            features['ela_std'] = float(np.std(gray_ela))
            features['ela_mean'] = float(np.mean(gray_ela))
            features['ela_max'] = float(np.max(gray_ela))
            
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
            features['log_hf_ratio'] = -10.0 if low_p <= 0 or high_p <= 0 else np.log10(high_p / low_p)
            
            # Multiple frequency zones
            for zone, start_pct in [('60', 0.6), ('70', 0.7), ('80', 0.8), ('85', 0.85), ('90', 0.9)]:
                start_idx = int(start_pct * n_freq)
                if len(psd1D[start_idx:]) > 5:
                    x_tail = np.log10(np.arange(start_idx, n_freq) + 1).reshape(-1, 1)
                    y_tail = np.log10(psd1D[start_idx:])
                    model = LinearRegression().fit(x_tail, y_tail)
                    features[f'tail_{zone}'] = float(model.coef_[0])
                else:
                    features[f'tail_{zone}'] = 0.0
            
            # Overall slope
            start_20 = int(0.2 * n_freq)
            if n_freq - start_20 > 20:
                x_full = np.log10(np.arange(start_20, n_freq) + 1).reshape(-1, 1)
                y_full = np.log10(psd1D[start_20:])
                model_full = LinearRegression().fit(x_full, y_full)
                features['spectral_slope'] = float(model_full.coef_[0])
            else:
                features['spectral_slope'] = 0.0

            # --- 3. Color Features ---
            color_feats = extract_color_features(im)
            features.update(color_feats)
            
            # --- 4. Noise Features ---
            noise_feats = extract_noise_features(img_gray)
            features.update(noise_feats)
            
            # --- 5. Wavelet ---
            wavelet_feats = extract_wavelet_features(img_gray)
            features.update(wavelet_feats)
            
            # --- 6. LBP ---
            lbp_feats = extract_lbp_features(img_gray)
            features.update(lbp_feats)
            
            # --- 7. Gradient ---
            grad_feats = extract_gradient_features(img_gray)
            features.update(grad_feats)

            return features
            
    except Exception as e:
        return None

def main():
    # === CONFIGURATION ===
    BASE_PATH = r"C:\Users\Vatase Radu\Downloads\datetrainingFACE"
    REAL_PATH = os.path.join(BASE_PATH, "training_real") 
    FAKE_PATH = os.path.join(BASE_PATH, "training_fake")
    
    print("🎯 ULTRA-OPTIMIZED AI DETECTION SYSTEM")
    print("="*80)
    print("Target: 90%+ Accuracy")
    print(f"📂 Dataset: {BASE_PATH}")
    
    if not os.path.exists(REAL_PATH):
        print(f"❌ Error: {REAL_PATH} not found")
        return

    # Index files
    real_files = [(f, "REAL") for f in glob.glob(os.path.join(REAL_PATH, "*.*"))]
    fake_files = [(f, "FAKE") for f in glob.glob(os.path.join(FAKE_PATH, "*.*"))]
    
    print(f"   Found: {len(real_files)} REAL, {len(fake_files)} FAKE")

    # Use MORE samples for better training
    SAMPLE_SIZE = 1500  # Increased from 1000
    if len(real_files) > SAMPLE_SIZE:
        print(f"⚡ Analyzing {SAMPLE_SIZE} images per class (total {SAMPLE_SIZE*2})")
        random.seed(42)
        random.shuffle(real_files)
        random.shuffle(fake_files)
        real_files = real_files[:SAMPLE_SIZE]
        fake_files = fake_files[:SAMPLE_SIZE]
    
    all_files = real_files + fake_files
    
    # Feature extraction
    print(f"\n🔬 Extracting ENHANCED features (40+ metrics)...")
    results = []
    with ProcessPoolExecutor() as executor:
        results_gen = list(tqdm(executor.map(analyze_image, all_files), total=len(all_files)))
    
    results = [r for r in results_gen if r is not None]
    
    if len(results) < 100:
        print("❌ Error: Too few images processed")
        return

    print(f"✅ Processed: {len(results)} images")
    
    # Save features
    csv_file = "optimized_features.csv"
    keys = results[0].keys()
    with open(csv_file, 'w', newline='') as f:
        dict_writer = csv.DictWriter(f, keys)
        dict_writer.writeheader()
        dict_writer.writerows(results)
    print(f"💾 Features saved: {csv_file}")

    # === PREPARE DATA ===
    print("\n" + "="*80)
    print("📊 DATA PREPARATION")
    print("="*80)
    
    feature_names = [k for k in results[0].keys() if k not in ['type', 'filename']]
    X = np.array([[r[f] for f in feature_names] for r in results])
    y = np.array([1 if r['type'] == 'FAKE' else 0 for r in results])
    
    print(f"Features: {len(feature_names)}")
    print(f"Samples: {len(X)} (REAL={np.sum(y==0)}, FAKE={np.sum(y==1)})")
    
    # Split with more test data for reliable evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # Feature Scaling (critical for SVM and Neural Networks)
    print("\n🔄 Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Feature Selection (keep top features)
    print("🔍 Selecting best features...")
    selector = SelectKBest(f_classif, k=min(35, len(feature_names)))  # Top 35 features
    X_train_selected = selector.fit_transform(X_train_scaled, y_train)
    X_test_selected = selector.transform(X_test_scaled)
    
    selected_indices = selector.get_support(indices=True)
    selected_features = [feature_names[i] for i in selected_indices]
    print(f"   Selected {len(selected_features)} best features")

    # === MODEL TRAINING & OPTIMIZATION ===
    print("\n" + "="*80)
    print("🤖 TRAINING MULTIPLE OPTIMIZED MODELS")
    print("="*80)
    
    models = {}
    
    # 1. Optimized Random Forest
    print("\n1️⃣ Random Forest (Hyperparameter Tuning)...")
    rf_params = {
        'n_estimators': [300, 500],
        'max_depth': [25, 30, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    rf_grid = GridSearchCV(rf, rf_params, cv=5, scoring='accuracy', n_jobs=-1, verbose=0)
    rf_grid.fit(X_train_selected, y_train)
    models['Random Forest'] = rf_grid.best_estimator_
    print(f"   Best params: {rf_grid.best_params_}")
    print(f"   CV Score: {rf_grid.best_score_:.4f}")
    
    # 2. XGBoost
    print("\n2️⃣ XGBoost (Gradient Boosting)...")
    xgb_params = {
        'n_estimators': [300, 500],
        'max_depth': [6, 8, 10],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.8, 1.0]
    }
    xgb_model = xgb.XGBClassifier(random_state=42, n_jobs=-1, use_label_encoder=False, eval_metric='logloss')
    xgb_grid = GridSearchCV(xgb_model, xgb_params, cv=5, scoring='accuracy', n_jobs=-1, verbose=0)
    xgb_grid.fit(X_train_selected, y_train)
    models['XGBoost'] = xgb_grid.best_estimator_
    print(f"   Best params: {xgb_grid.best_params_}")
    print(f"   CV Score: {xgb_grid.best_score_:.4f}")
    
    # 3. SVM
    print("\n3️⃣ Support Vector Machine...")
    svm_params = {
        'C': [0.1, 1, 10],
        'gamma': ['scale', 'auto'],
        'kernel': ['rbf', 'poly']
    }
    svm_model = SVC(probability=True, random_state=42)
    svm_grid = GridSearchCV(svm_model, svm_params, cv=5, scoring='accuracy', n_jobs=-1, verbose=0)
    svm_grid.fit(X_train_selected, y_train)
    models['SVM'] = svm_grid.best_estimator_
    print(f"   Best params: {svm_grid.best_params_}")
    print(f"   CV Score: {svm_grid.best_score_:.4f}")
    
    # 4. Neural Network
    print("\n4️⃣ Neural Network (MLP)...")
    mlp_params = {
        'hidden_layer_sizes': [(100, 50), (150, 75), (200, 100)],
        'alpha': [0.0001, 0.001],
        'learning_rate_init': [0.001, 0.01]
    }
    mlp_model = MLPClassifier(max_iter=500, random_state=42)
    mlp_grid = GridSearchCV(mlp_model, mlp_params, cv=5, scoring='accuracy', n_jobs=-1, verbose=0)
    mlp_grid.fit(X_train_selected, y_train)
    models['Neural Network'] = mlp_grid.best_estimator_
    print(f"   Best params: {mlp_grid.best_params_}")
    print(f"   CV Score: {mlp_grid.best_score_:.4f}")
    
    # 5. Ensemble Voting
    print("\n5️⃣ Ensemble (Voting Classifier)...")
    voting_clf = VotingClassifier(
        estimators=[
            ('rf', models['Random Forest']),
            ('xgb', models['XGBoost']),
            ('svm', models['SVM']),
            ('mlp', models['Neural Network'])
        ],
        voting='soft',
        n_jobs=-1
    )
    voting_clf.fit(X_train_selected, y_train)
    models['Ensemble'] = voting_clf
    
    # === EVALUATION ===
    print("\n" + "="*80)
    print("📈 FINAL RESULTS - ALL MODELS")
    print("="*80)
    
    results_summary = []
    best_model_name = None
    best_accuracy = 0
    
    for name, model in models.items():
        y_pred = model.predict(X_test_selected)
        y_proba = model.predict_proba(X_test_selected)[:, 1] if hasattr(model, 'predict_proba') else y_pred
        
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        cm = confusion_matrix(y_test, y_pred)
        
        # Precision and Recall
        precision_fake = cm[1,1] / (cm[1,1] + cm[0,1]) if (cm[1,1] + cm[0,1]) > 0 else 0
        recall_fake = cm[1,1] / (cm[1,1] + cm[1,0]) if (cm[1,1] + cm[1,0]) > 0 else 0
        
        results_summary.append({
            'name': name,
            'accuracy': accuracy,
            'auc': auc,
            'cm': cm,
            'precision': precision_fake,
            'recall': recall_fake
        })
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_name = name
        
        print(f"\n{'='*80}")
        print(f"🏆 {name}")
        print(f"{'='*80}")
        print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"ROC AUC:   {auc:.4f}")
        print(f"Precision: {precision_fake:.4f}")
        print(f"Recall:    {recall_fake:.4f}")
        print(f"\nConfusion Matrix:")
        print(f"   TN={cm[0,0]:4d} | FP={cm[0,1]:4d}")
        print(f"   FN={cm[1,0]:4d} | TP={cm[1,1]:4d}")
    
    # === BEST MODEL ANALYSIS ===
    print("\n" + "="*80)
    print(f"🥇 BEST MODEL: {best_model_name}")
    print(f"   Accuracy: {best_accuracy*100:.2f}%")
    print("="*80)
    
    best_model = models[best_model_name]
    best_result = [r for r in results_summary if r['name'] == best_model_name][0]
    
    # Save best model
    with open('best_model.pkl', 'wb') as f:
        pickle.dump({
            'model': best_model,
            'scaler': scaler,
            'selector': selector,
            'feature_names': feature_names
        }, f)
    print("💾 Best model saved: best_model.pkl")
    
    # === PLOTTING ===
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Model comparison
    ax1 = fig.add_subplot(gs[0, :])
    model_names = [r['name'] for r in results_summary]
    accuracies = [r['accuracy'] * 100 for r in results_summary]
    colors = ['gold' if r['name'] == best_model_name else 'steelblue' for r in results_summary]
    bars = ax1.bar(model_names, accuracies, color=colors, edgecolor='black', linewidth=1.5)
    ax1.axhline(y=90, color='red', linestyle='--', linewidth=2, label='Target: 90%')
    ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Model Comparison - Accuracy', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold')
    
    # ROC Curves
    ax2 = fig.add_subplot(gs[1, 0])
    for result in results_summary:
        model = models[result['name']]
        y_proba = model.predict_proba(X_test_selected)[:, 1] if hasattr(model, 'predict_proba') else model.predict(X_test_selected)
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        ax2.plot(fpr, tpr, linewidth=2, label=f"{result['name']} (AUC={result['auc']:.3f})")
    ax2.plot([0, 1], [0, 1], 'k--', label='Random')
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('ROC Curves - All Models')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # Best Model Confusion Matrix
    ax3 = fig.add_subplot(gs[1, 1])
    cm = best_result['cm']
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    im = ax3.imshow(cm_normalized, cmap='Blues', aspect='auto')
    ax3.set_xticks([0, 1])
    ax3.set_yticks([0, 1])
    ax3.set_xticklabels(['REAL', 'FAKE'])
    ax3.set_yticklabels(['REAL', 'FAKE'])
    ax3.set_xlabel('Predicted')
    ax3.set_ylabel('Actual')
    ax3.set_title(f'Best Model: {best_model_name}')
    for i in range(2):
        for j in range(2):
            text = ax3.text(j, i, f'{cm_normalized[i, j]:.2f}\n({cm[i, j]})',
                           ha="center", va="center", color="black", fontweight='bold')
    
    # Metrics comparison
    ax4 = fig.add_subplot(gs[1, 2])
    metrics = ['Accuracy', 'AUC', 'Precision', 'Recall']
    x = np.arange(len(metrics))
    width = 0.15
    for i, result in enumerate(results_summary):
        values = [result['accuracy'], result['auc'], result['precision'], result['recall']]
        ax4.bar(x + i*width, values, width, label=result['name'])
    ax4.set_ylabel('Score')
    ax4.set_title('Metrics Comparison')
    ax4.set_xticks(x + width * 2)
    ax4.set_xticklabels(metrics)
    ax4.legend(fontsize=8)
    ax4.grid(axis='y', alpha=0.3)
    
    # Feature importance (if available)
    if hasattr(best_model, 'feature_importances_'):
        ax5 = fig.add_subplot(gs[2, :])
        importances = best_model.feature_importances_
        indices = np.argsort(importances)[::-1][:15]
        ax5.barh(range(15), importances[indices], color='steelblue')
        ax5.set_yticks(range(15))
        ax5.set_yticklabels([selected_features[i] for i in indices])
        ax5.set_xlabel('Importance')
        ax5.set_title(f'Top 15 Features - {best_model_name}')
        ax5.invert_yaxis()
    
    plt.suptitle('ULTRA-OPTIMIZED AI DETECTION - COMPLETE ANALYSIS', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    plt.savefig("optimized_results.png", dpi=150, bbox_inches='tight')
    print("📊 Analysis saved: optimized_results.png")
    
    # Final Summary
    print("\n" + "="*80)
    print("✨ FINAL SUMMARY")
    print("="*80)
    print(f"🎯 Target Accuracy: 90%")
    print(f"🏆 Best Achieved:   {best_accuracy*100:.2f}% ({best_model_name})")
    if best_accuracy >= 0.9:
        print("✅ TARGET REACHED! 🎉")
    else:
        gap = 90 - best_accuracy*100
        print(f"📊 Gap to target:   {gap:.2f}%")
        print("\n💡 Recommendations:")
        print("   - Dataset might contain very high-quality AI images")
        print("   - Consider adding PRNU (camera sensor noise) analysis")
        print("   - Deep learning (CNN) might achieve better results")
    print("="*80)
    
    plt.show()

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
