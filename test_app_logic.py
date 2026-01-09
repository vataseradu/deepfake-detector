"""
Quick test of the updated app logic with real image
"""

import os
import numpy as np
from PIL import Image
import cv2
from scipy.fft import fft2, fftshift
from frequency import azimuthalAverage
import pickle

# Load model
with open('face_rf_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)

REAL_IMAGE = r"/path/to/dataset/training_real/sample.jpg"  # TODO: Set your path

img = Image.open(REAL_IMAGE).convert('RGB')
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

features_dict = {
    'tail_90': np.gradient(psd1D)[int(0.9*psd_len):].mean() if psd_len > 50 else 0,
    'tail_80': np.gradient(psd1D)[int(0.8*psd_len):].mean() if psd_len > 50 else 0,
    'tail_70': np.gradient(psd1D)[int(0.7*psd_len):].mean() if psd_len > 50 else 0,
    'mean_power': np.mean(psd1D),
    'std_power': np.std(psd1D),
    'hf_lf_ratio': (np.mean(psd1D[int(0.7*psd_len):]) / 
                   (np.mean(psd1D[:int(0.4*psd_len)]) + 1e-10)) if psd_len > 50 else 0
}

print("Features extracted from REAL image:")
print(f"  tail_70: {features_dict['tail_70']:.6f}")
print(f"  tail_80: {features_dict['tail_80']:.6f}")
print(f"  tail_90: {features_dict['tail_90']:.6f}")
print(f"  hf_lf_ratio: {features_dict['hf_lf_ratio']:.6f}")
print(f"  std_power: {features_dict['std_power']:.2f}")

# Predict using RF model
X_features = np.array([[
    features_dict['tail_70'],
    features_dict['tail_80'],
    features_dict['tail_90'],
    features_dict['hf_lf_ratio'],
    features_dict['std_power']
]])

proba = rf_model.predict_proba(X_features)[0]
math_score_ai = proba[1] * 100

print(f"\n🎯 Random Forest Prediction:")
print(f"  P(REAL): {proba[0]*100:.1f}%")
print(f"  P(FAKE): {proba[1]*100:.1f}%")
print(f"\nFinal AI Score: {math_score_ai:.1f}%")
print(f"Verdict: {'AI-GENERATED' if math_score_ai > 60 else 'REAL'}")
