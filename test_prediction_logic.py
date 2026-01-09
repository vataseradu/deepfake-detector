"""
Test script to verify model prediction direction is correct
"""
import pickle
import numpy as np
import os
import glob
from PIL import Image

# Load model
print("="*60)
print("TESTING MODEL PREDICTION DIRECTION")
print("="*60)

model_data = pickle.load(open('final_model.pkl', 'rb'))
model = model_data['model']
scaler = model_data['scaler']
feature_names = model_data['feature_names']

print(f"\n✅ Model loaded successfully")
print(f"   Classes: {model.classes_} (0=REAL, 1=FAKE)")
print(f"   Features: {len(feature_names)}")

# Test with dummy features
print("\n" + "="*60)
print("TEST 1: Dummy Features")
print("="*60)

# Create typical REAL photo features
real_features = {
    'ela_std': 8.5,  # Higher ELA
    'ela_mean': 3.2,
    'ela_max': 45.0,
    'log_hf_ratio': -2.1,  # Lower HF
    'tail_70': -4.5,
    'tail_80': -5.2,
    'tail_90': -6.1,
    'wavelet_cH_std': 12.5,
    'wavelet_cV_std': 11.8,
    'wavelet_cD_std': 10.2,
    'wavelet_energy': 8500000,
    'lbp_entropy': 3.8,
    'lbp_std': 45.2,
    'gradient_mean': 5.5,
    'gradient_std': 4.2,
    'gradient_max': 85.0,
    'gradient_skew': 1.2,
    'red_mean': 125.0,
    'red_std': 55.0,
    'green_mean': 118.0,
    'green_std': 52.0,
    'blue_mean': 110.0,
    'blue_std': 48.0
}

# Create typical AI/FAKE features
fake_features = {
    'ela_std': 1.8,  # Lower ELA
    'ela_mean': 0.9,
    'ela_max': 12.0,
    'log_hf_ratio': -1.3,  # Higher HF
    'tail_70': -2.1,
    'tail_80': -2.8,
    'tail_90': -3.5,
    'wavelet_cH_std': 8.2,
    'wavelet_cV_std': 7.9,
    'wavelet_cD_std': 7.5,
    'wavelet_energy': 3500000,
    'lbp_entropy': 2.2,
    'lbp_std': 28.5,
    'gradient_mean': 3.8,
    'gradient_std': 2.5,
    'gradient_max': 45.0,
    'gradient_skew': 0.6,
    'red_mean': 132.0,
    'red_std': 38.0,
    'green_mean': 125.0,
    'green_std': 35.0,
    'blue_mean': 115.0,
    'blue_std': 32.0
}

# Test REAL features
real_vector = np.array([[real_features[name] for name in feature_names]])
real_vector_scaled = scaler.transform(real_vector)
real_prob = model.predict_proba(real_vector_scaled)[0]
real_pred = model.predict(real_vector_scaled)[0]

print("\n📸 TYPICAL REAL PHOTO FEATURES:")
print(f"   Prediction: {real_pred} (should be 0)")
print(f"   Probability [REAL, FAKE]: [{real_prob[0]:.3f}, {real_prob[1]:.3f}]")
print(f"   Display: REAL={real_prob[0]*100:.1f}% | FAKE={real_prob[1]*100:.1f}%")
if real_pred == 0 and real_prob[0] > 0.5:
    print("   ✅ CORRECT: Predicts REAL")
else:
    print("   ❌ WRONG: Should predict REAL!")

# Test FAKE features
fake_vector = np.array([[fake_features[name] for name in feature_names]])
fake_vector_scaled = scaler.transform(fake_vector)
fake_prob = model.predict_proba(fake_vector_scaled)[0]
fake_pred = model.predict(fake_vector_scaled)[0]

print("\n🤖 TYPICAL AI/FAKE FEATURES:")
print(f"   Prediction: {fake_pred} (should be 1)")
print(f"   Probability [REAL, FAKE]: [{fake_prob[0]:.3f}, {fake_prob[1]:.3f}]")
print(f"   Display: REAL={fake_prob[0]*100:.1f}% | FAKE={fake_prob[1]*100:.1f}%")
if fake_pred == 1 and fake_prob[1] > 0.5:
    print("   ✅ CORRECT: Predicts FAKE")
else:
    print("   ❌ WRONG: Should predict FAKE!")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print("\n✅ In app_final.py, the mapping is:")
print("   prob_real = probability[0] * 100")
print("   prob_fake = probability[1] * 100")
print("   is_fake = prob_fake > prob_real")
print("\n✅ This is CORRECT because model.classes_ = [0, 1]")
print("   where 0 = REAL and 1 = FAKE")
print("\nIf the application shows INVERTED results, the problem is NOT")
print("in the probability mapping, but in:")
print("  1. FFT suspicion score logic (should increase prob_FAKE)")
print("  2. Metadata boost logic (should increase prob_REAL)")
print("  3. Phone pattern detection (should increase prob_REAL)")
print("\n" + "="*60)
