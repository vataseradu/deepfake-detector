"""
Calibrare threshold-uri FFT pe dataset FACE (training_fake + training_real)
960 fake images + 1081 real images = 2041 total
"""

import os
import numpy as np
from PIL import Image
import cv2
from scipy.fft import fft2, fftshift
from frequency import azimuthalAverage
import pandas as pd
from tqdm import tqdm

# Paths
DATASET_PATH = r"C:\Users\Vatase Radu\Downloads\datetrainingFACE"
FAKE_PATH = os.path.join(DATASET_PATH, "training_fake")
REAL_PATH = os.path.join(DATASET_PATH, "training_real")

def compute_fft_features(img_path):
    """Calculeaza features FFT pentru o imagine"""
    try:
        img = Image.open(img_path).convert('RGB')
        img_array = np.array(img)
        
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        f = fft2(gray)
        fshift = fftshift(f)
        magnitude_spectrum = np.abs(fshift)
        magnitude_spectrum = np.log1p(magnitude_spectrum)
        
        psd1D = azimuthalAverage(magnitude_spectrum)
        
        if psd1D is None or len(psd1D) < 50:
            return None
        
        psd_len = len(psd1D)
        
        # Features
        features = {}
        
        # Tail gradients
        for pct in [0.7, 0.8, 0.9]:
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
        
        # HF/LF ratio
        cutoff = len(psd1D) // 2
        lf_power = np.mean(psd1D[:cutoff])
        hf_power = np.mean(psd1D[cutoff:])
        features['hf_lf_ratio'] = hf_power / lf_power if lf_power > 0 else 0
        
        # Power stats
        features['mean_power'] = np.mean(psd1D)
        features['std_power'] = np.std(psd1D)
        features['power_range'] = np.ptp(psd1D[np.isfinite(psd1D)])
        
        # Decay linearity
        mid_idx = len(psd1D) // 2
        x = np.arange(mid_idx, len(psd1D))
        y = psd1D[mid_idx:]
        valid = np.isfinite(y)
        if np.sum(valid) > 10:
            from scipy.stats import pearsonr
            corr, _ = pearsonr(x[valid], y[valid])
            features['decay_linearity'] = corr
        else:
            features['decay_linearity'] = 0
        
        return features
    
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None

def analyze_dataset():
    """Analizeaza tot dataset-ul si gaseste threshold-uri optime"""
    print("Analyzing FACE dataset...")
    print(f"Fake images: {len(os.listdir(FAKE_PATH))}")
    print(f"Real images: {len(os.listdir(REAL_PATH))}")
    
    results = []
    
    # Process FAKE images
    print("\nProcessing FAKE images...")
    fake_files = [f for f in os.listdir(FAKE_PATH) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    for filename in tqdm(fake_files[:960]):  # All 960 fake images
        img_path = os.path.join(FAKE_PATH, filename)
        features = compute_fft_features(img_path)
        if features:
            features['label'] = 'FAKE'
            features['filename'] = filename
            results.append(features)
    
    # Process REAL images
    print("\nProcessing REAL images...")
    real_files = [f for f in os.listdir(REAL_PATH) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    for filename in tqdm(real_files[:1081]):  # All 1081 real images
        img_path = os.path.join(REAL_PATH, filename)
        features = compute_fft_features(img_path)
        if features:
            features['label'] = 'REAL'
            features['filename'] = filename
            results.append(features)
    
    df = pd.DataFrame(results)
    
    print(f"\nTotal processed: {len(df)} images")
    print(f"  - FAKE: {len(df[df['label'] == 'FAKE'])}")
    print(f"  - REAL: {len(df[df['label'] == 'REAL'])}")
    
    # Save raw data
    df.to_csv('face_dataset_calibration.csv', index=False)
    print("\nSaved: face_dataset_calibration.csv")
    
    # Analyze features
    print("\n" + "="*60)
    print("FEATURE STATISTICS")
    print("="*60)
    
    for feature in ['tail_70', 'tail_80', 'tail_90', 'hf_lf_ratio', 'decay_linearity']:
        fake_vals = df[df['label'] == 'FAKE'][feature]
        real_vals = df[df['label'] == 'REAL'][feature]
        
        print(f"\n{feature}:")
        print(f"  FAKE: mean={fake_vals.mean():.4f}, std={fake_vals.std():.4f}")
        print(f"  REAL: mean={real_vals.mean():.4f}, std={real_vals.std():.4f}")
        print(f"  Separation: {abs(fake_vals.mean() - real_vals.mean()):.4f}")
    
    # Find optimal thresholds
    print("\n" + "="*60)
    print("OPTIMAL THRESHOLDS")
    print("="*60)
    
    best_accuracy = 0
    best_thresholds = {}
    
    # Test different threshold combinations
    for tail_90_thresh in np.arange(-3.0, 0.0, 0.1):
        for hf_lf_thresh in np.arange(0.2, 0.8, 0.05):
            correct = 0
            total = 0
            
            for _, row in df.iterrows():
                suspicion = 0
                
                # Tail 90 check
                if row['tail_90'] > tail_90_thresh:
                    suspicion += 35
                
                # HF/LF ratio check
                if row['hf_lf_ratio'] > hf_lf_thresh:
                    suspicion += 30
                
                # Tail 80 check
                if row['tail_80'] > -2.0:
                    suspicion += 20
                
                # Decay linearity
                if abs(row['decay_linearity']) < 0.7:
                    suspicion += 15
                
                predicted = 'FAKE' if suspicion > 50 else 'REAL'
                if predicted == row['label']:
                    correct += 1
                total += 1
            
            accuracy = correct / total
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_thresholds = {
                    'tail_90': tail_90_thresh,
                    'hf_lf_ratio': hf_lf_thresh,
                    'accuracy': accuracy
                }
    
    print(f"\nBest accuracy: {best_accuracy*100:.2f}%")
    print(f"Optimal thresholds:")
    for key, val in best_thresholds.items():
        print(f"  {key}: {val:.4f}")
    
    # Test with best thresholds
    print("\n" + "="*60)
    print("CONFUSION MATRIX")
    print("="*60)
    
    tp = fp = tn = fn = 0
    
    for _, row in df.iterrows():
        suspicion = 0
        
        if row['tail_90'] > best_thresholds['tail_90']:
            suspicion += 35
        if row['hf_lf_ratio'] > best_thresholds['hf_lf_ratio']:
            suspicion += 30
        if row['tail_80'] > -2.0:
            suspicion += 20
        if abs(row['decay_linearity']) < 0.7:
            suspicion += 15
        
        predicted = 'FAKE' if suspicion > 50 else 'REAL'
        
        if row['label'] == 'FAKE' and predicted == 'FAKE':
            tp += 1
        elif row['label'] == 'FAKE' and predicted == 'REAL':
            fn += 1
        elif row['label'] == 'REAL' and predicted == 'REAL':
            tn += 1
        elif row['label'] == 'REAL' and predicted == 'FAKE':
            fp += 1
    
    print(f"\nTrue Positives (FAKE detected as FAKE): {tp}")
    print(f"False Negatives (FAKE detected as REAL): {fn}")
    print(f"True Negatives (REAL detected as REAL): {tn}")
    print(f"False Positives (REAL detected as FAKE): {fp}")
    print(f"\nPrecision: {tp/(tp+fp)*100:.2f}%")
    print(f"Recall: {tp/(tp+fn)*100:.2f}%")
    print(f"Accuracy: {(tp+tn)/(tp+tn+fp+fn)*100:.2f}%")
    
    return best_thresholds

if __name__ == "__main__":
    thresholds = analyze_dataset()
    
    print("\n" + "="*60)
    print("RECOMMENDED UPDATES FOR decision_logic.py:")
    print("="*60)
    print(f"""
def detect_fft_patterns(features_dict):
    suspicion_score = 0
    patterns = {{}}
    
    # Tail 90 threshold (optimized)
    if features_dict['tail_90'] > {thresholds['tail_90']:.3f}:
        suspicion_score += 35
        patterns['flat_tail'] = True
    
    # HF/LF ratio threshold (optimized)
    if features_dict['hf_lf_ratio'] > {thresholds['hf_lf_ratio']:.3f}:
        suspicion_score += 30
        patterns['high_freq_anomaly'] = True
    
    # Tail 80 check
    if features_dict['tail_80'] > -2.0:
        suspicion_score += 20
        patterns['unnatural_decay'] = True
    
    # Decay linearity
    if abs(features_dict['decay_linearity']) < 0.7:
        suspicion_score += 15
        patterns['non_linear'] = True
    
    patterns['suspicion_score'] = suspicion_score
    return patterns

# Expected accuracy: {thresholds['accuracy']*100:.2f}%
""")
