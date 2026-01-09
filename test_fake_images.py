"""
Test rapid pe imagini FAKE pentru a vedea ce valori primesc
"""

import os
import numpy as np
from PIL import Image
import cv2
from scipy.fft import fft2, fftshift
from frequency import azimuthalAverage

FAKE_PATH = r"/path/to/dataset/training_fake"  # TODO: Set your path (see SETUP_LOCAL.md)

def analyze_image(img_path):
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
    
    # Tail gradients
    tail_90_idx = int(0.9 * psd_len)
    tail_90_end = min(tail_90_idx + 10, psd_len)
    x = np.arange(tail_90_idx, tail_90_end)
    y = psd1D[tail_90_idx:tail_90_end]
    valid = np.isfinite(y)
    if np.sum(valid) > 1:
        coeffs = np.polyfit(x[valid], y[valid], 1)
        tail_90 = coeffs[0]
    else:
        tail_90 = 0
    
    # HF/LF ratio
    cutoff = len(psd1D) // 2
    lf_power = np.mean(psd1D[:cutoff])
    hf_power = np.mean(psd1D[cutoff:])
    hf_lf_ratio = hf_power / lf_power if lf_power > 0 else 0
    
    return {
        'tail_90': tail_90,
        'hf_lf_ratio': hf_lf_ratio,
        'mean_power': np.mean(psd1D),
        'std_power': np.std(psd1D)
    }

# Test 10 imagini FAKE
print("Testing 10 FAKE images:")
print("="*60)

fake_files = [f for f in os.listdir(FAKE_PATH) if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:10]

for filename in fake_files:
    img_path = os.path.join(FAKE_PATH, filename)
    features = analyze_image(img_path)
    if features:
        print(f"\n{filename}:")
        print(f"  tail_90: {features['tail_90']:.6f}")
        print(f"  hf_lf_ratio: {features['hf_lf_ratio']:.6f}")
        print(f"  mean_power: {features['mean_power']:.2f}")
        
        # Simulare suspicion score (logica VECHE greșită)
        suspicion = 0
        if features['tail_90'] > -1.0:
            suspicion += 35
            print(f"  -> FLAT TAIL (tail_90 > -1.0): +35")
        if features['hf_lf_ratio'] > 0.35:
            suspicion += 30
            print(f"  -> HIGH HF/LF (> 0.35): +30")
        
        print(f"  TOTAL SUSPICION: {suspicion}/100")
        print(f"  AI SCORE: {suspicion}%")
