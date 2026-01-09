"""
Batch test of the updated classification logic
Tests 20 REAL + 20 FAKE images
"""

import os
import numpy as np
from PIL import Image
import cv2
from scipy.fft import fft2, fftshift
from frequency import azimuthalAverage
import pickle

# Load model
with open('face_rf_simple.pkl', 'rb') as f:
    rf_model = pickle.load(f)

FAKE_PATH = r"/path/to/dataset/training_fake"  # TODO: Set your path (see SETUP_LOCAL.md)
REAL_PATH = r"/path/to/dataset/training_real"  # TODO: Set your path (see SETUP_LOCAL.md)

def analyze_image(img_path):
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
    
    features_dict = {
        'tail_90': np.gradient(psd1D)[int(0.9*psd_len):].mean() if psd_len > 50 else 0,
        'tail_80': np.gradient(psd1D)[int(0.8*psd_len):].mean() if psd_len > 50 else 0,
        'tail_70': np.gradient(psd1D)[int(0.7*psd_len):].mean() if psd_len > 50 else 0,
        'hf_lf_ratio': (np.mean(psd1D[int(0.7*psd_len):]) / 
                       (np.mean(psd1D[:int(0.4*psd_len)]) + 1e-10)) if psd_len > 50 else 0,
        'std_power': np.std(psd1D)
    }
    
    X_features = np.array([[
        features_dict['tail_70'],
        features_dict['tail_80'],
        features_dict['tail_90'],
        features_dict['hf_lf_ratio'],
        features_dict['std_power']
    ]])
    
    proba = rf_model.predict_proba(X_features)[0]
    ai_score = proba[1] * 100
    verdict = 'AI' if ai_score > 60 else 'REAL'
    
    return ai_score, verdict

print("="*70)
print("BATCH TEST: REAL vs FAKE Classification")
print("="*70)

# Test REAL images
print("\n📷 Testing 20 REAL images:")
print("-"*70)

real_files = [f for f in os.listdir(REAL_PATH) if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:20]

real_correct = 0
real_scores = []

for i, filename in enumerate(real_files, 1):
    img_path = os.path.join(REAL_PATH, filename)
    ai_score, verdict = analyze_image(img_path)
    real_scores.append(ai_score)
    
    status = "✅" if verdict == "REAL" else "❌"
    
    if verdict == "REAL":
        real_correct += 1
    
    print(f"{i:2d}. {filename:25s} | AI: {ai_score:5.1f}% | {verdict:4s} {status}")

print(f"\n📊 REAL Accuracy: {real_correct}/20 = {real_correct/20*100:.1f}%")
print(f"   Average AI score: {np.mean(real_scores):.1f}%")
print(f"   Min: {np.min(real_scores):.1f}%, Max: {np.max(real_scores):.1f}%")

# Test FAKE images
print("\n🤖 Testing 20 FAKE images:")
print("-"*70)

fake_files = [f for f in os.listdir(FAKE_PATH) if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:20]

fake_correct = 0
fake_scores = []

for i, filename in enumerate(fake_files, 1):
    img_path = os.path.join(FAKE_PATH, filename)
    ai_score, verdict = analyze_image(img_path)
    fake_scores.append(ai_score)
    
    status = "✅" if verdict == "AI" else "❌"
    
    if verdict == "AI":
        fake_correct += 1
    
    print(f"{i:2d}. {filename:25s} | AI: {ai_score:5.1f}% | {verdict:4s} {status}")

print(f"\n📊 FAKE Accuracy: {fake_correct}/20 = {fake_correct/20*100:.1f}%")
print(f"   Average AI score: {np.mean(fake_scores):.1f}%")
print(f"   Min: {np.min(fake_scores):.1f}%, Max: {np.max(fake_scores):.1f}%")

# Overall stats
print("\n" + "="*70)
print("OVERALL RESULTS")
print("="*70)

total_correct = real_correct + fake_correct
total_tests = 40

print(f"✅ Total Correct: {total_correct}/40 = {total_correct/40*100:.1f}%")
print(f"❌ Total Wrong:   {40-total_correct}/40 = {(40-total_correct)/40*100:.1f}%")
print(f"\n📈 Distribution:")
print(f"   REAL average: {np.mean(real_scores):.1f}% AI")
print(f"   FAKE average: {np.mean(fake_scores):.1f}% AI")
print(f"   Separation: {abs(np.mean(fake_scores) - np.mean(real_scores)):.1f}%")

if abs(np.mean(fake_scores) - np.mean(real_scores)) > 15:
    print("\n✅ Good separation between REAL and FAKE!")
else:
    print("\n⚠️ Poor separation - FFT features struggle with this dataset")
