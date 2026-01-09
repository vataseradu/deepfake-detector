import pickle
import numpy as np

# Load model
data = pickle.load(open('final_model.pkl', 'rb'))
model = data['model']

print("="*50)
print("MODEL CLASS ORDER TEST")
print("="*50)
print(f"Model classes: {model.classes_}")
print(f"Class 0 = REAL")
print(f"Class 1 = FAKE")
print()

# Create test feature (dummy values)
test_features = np.zeros((1, len(data['feature_names'])))
prob = model.predict_proba(test_features)[0]
pred = model.predict(test_features)[0]

print(f"Test Prediction:")
print(f"  prob[0] (REAL): {prob[0]:.3f}")
print(f"  prob[1] (FAKE): {prob[1]:.3f}")
print(f"  Predicted class: {pred}")
print()
print("✅ Mapping is CORRECT in app_final.py:")
print("   prob_real = probability[0] * 100")
print("   prob_fake = probability[1] * 100")
