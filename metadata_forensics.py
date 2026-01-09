"""
CYBERSECURITY METADATA FORENSICS FOR AI IMAGE DETECTION
========================================================
Digital Forensics Approach - Metadata Analysis

Features:
1. EXIF Data Extraction (Camera, Software, GPS, Timestamps)
2. AI Tool Signature Detection (Photoshop, DALL-E, Midjourney, etc.)
3. Metadata Inconsistency Analysis
4. Timestamp Anomaly Detection
5. Digital Fingerprinting
6. Combined ML Classification

Cybersecurity Relevance:
- Digital Evidence Authentication
- Deepfake Detection
- Image Tampering Investigation
- Chain of Custody Verification
"""

import os
import glob
import csv
import random
import re
import pickle
from datetime import datetime
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import piexif
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# AI Tool Signatures (known fingerprints)
AI_SIGNATURES = {
    'midjourney': ['midjourney', 'mj', 'discord'],
    'dalle': ['dall-e', 'dall·e', 'openai'],
    'stable_diffusion': ['stable diffusion', 'sd', 'automatic1111', 'comfyui'],
    'photoshop_ai': ['adobe photoshop', 'generative fill', 'neural filters'],
    'canva': ['canva', 'canva.com'],
    'firefly': ['adobe firefly'],
    'leonardo': ['leonardo.ai'],
    'clipdrop': ['clipdrop'],
    'playground': ['playground ai']
}

class MetadataForensics:
    """Extract and analyze image metadata for AI detection"""
    
    def __init__(self):
        self.suspicious_keywords = list(AI_SIGNATURES.keys())
        
    def extract_exif(self, img_path):
        """Extract all EXIF data from image"""
        try:
            img = Image.open(img_path)
            exif_data = {}
            
            # Try to get EXIF
            if hasattr(img, '_getexif') and img._getexif():
                exif = img._getexif()
                for tag_id, value in exif.items():
                    tag = TAGS.get(tag_id, tag_id)
                    exif_data[tag] = value
            
            # Try piexif for more complete data
            try:
                exif_dict = piexif.load(img.info.get('exif', b''))
                for ifd in exif_dict:
                    if ifd == "thumbnail":
                        continue
                    for tag in exif_dict[ifd]:
                        tag_name = piexif.TAGS[ifd][tag]["name"]
                        exif_data[tag_name] = exif_dict[ifd][tag]
            except:
                pass
            
            return exif_data
        except:
            return {}
    
    def detect_ai_signature(self, exif_data, img_path):
        """Detect AI tool signatures in metadata"""
        detected_tools = []
        confidence_scores = {}
        
        # Check all metadata fields
        all_text = ' '.join([str(v).lower() for v in exif_data.values()])
        
        # Check filename
        filename = os.path.basename(img_path).lower()
        all_text += ' ' + filename
        
        # Search for signatures
        for tool, keywords in AI_SIGNATURES.items():
            matches = sum(1 for keyword in keywords if keyword in all_text)
            if matches > 0:
                detected_tools.append(tool)
                confidence_scores[tool] = matches / len(keywords)
        
        return detected_tools, confidence_scores
    
    def analyze_metadata_completeness(self, exif_data):
        """Analyze metadata completeness (AI images often lack camera data)"""
        critical_fields = [
            'Make', 'Model',  # Camera manufacturer and model
            'DateTime', 'DateTimeOriginal',  # Timestamps
            'Software',  # Processing software
            'ExposureTime', 'FNumber', 'ISOSpeedRatings',  # Camera settings
            'FocalLength',  # Lens info
            'Flash',  # Flash usage
            'Orientation'  # Image orientation
        ]
        
        present_fields = [field for field in critical_fields if field in exif_data]
        completeness_score = len(present_fields) / len(critical_fields)
        
        return {
            'completeness': completeness_score,
            'present_count': len(present_fields),
            'missing_count': len(critical_fields) - len(present_fields),
            'has_camera_info': any(f in exif_data for f in ['Make', 'Model']),
            'has_timestamp': any(f in exif_data for f in ['DateTime', 'DateTimeOriginal']),
            'has_software': 'Software' in exif_data,
            'has_camera_settings': any(f in exif_data for f in ['ExposureTime', 'FNumber', 'ISOSpeedRatings'])
        }
    
    def detect_timestamp_anomalies(self, exif_data):
        """Detect suspicious timestamp patterns"""
        anomalies = []
        
        try:
            # Get timestamps
            dt_original = exif_data.get('DateTimeOriginal', '')
            dt_digitized = exif_data.get('DateTimeDigitized', '')
            dt_modified = exif_data.get('DateTime', '')
            
            # Check for missing timestamps
            if not dt_original:
                anomalies.append('missing_original_timestamp')
            
            # Check for unrealistic dates (e.g., future dates)
            current_year = datetime.now().year
            for dt_str in [dt_original, dt_digitized, dt_modified]:
                if dt_str:
                    try:
                        # Extract year from timestamp (format: YYYY:MM:DD HH:MM:SS)
                        year = int(str(dt_str)[:4])
                        if year > current_year:
                            anomalies.append('future_timestamp')
                        if year < 1990:  # Digital cameras weren't common before 1990
                            anomalies.append('unrealistic_old_timestamp')
                    except:
                        pass
            
            # Check for identical timestamps (suspicious for real photos)
            timestamps = [dt_original, dt_digitized, dt_modified]
            if len(timestamps) == len(set(timestamps)) == 1 and timestamps[0]:
                anomalies.append('identical_timestamps')
                
        except:
            pass
        
        return anomalies
    
    def extract_features(self, img_path):
        """Extract all metadata forensic features"""
        features = {}
        
        # Get EXIF data
        exif_data = self.extract_exif(img_path)
        
        # 1. AI Signature Detection
        detected_tools, confidence = self.detect_ai_signature(exif_data, img_path)
        features['has_ai_signature'] = 1 if detected_tools else 0
        features['ai_confidence'] = max(confidence.values()) if confidence else 0.0
        features['ai_tool_count'] = len(detected_tools)
        
        # 2. Metadata Completeness
        completeness = self.analyze_metadata_completeness(exif_data)
        features['metadata_completeness'] = completeness['completeness']
        features['has_camera_info'] = 1 if completeness['has_camera_info'] else 0
        features['has_timestamp'] = 1 if completeness['has_timestamp'] else 0
        features['has_software'] = 1 if completeness['has_software'] else 0
        features['has_camera_settings'] = 1 if completeness['has_camera_settings'] else 0
        features['missing_fields_count'] = completeness['missing_count']
        
        # 3. Timestamp Anomalies
        anomalies = self.detect_timestamp_anomalies(exif_data)
        features['timestamp_anomaly_count'] = len(anomalies)
        features['has_missing_timestamp'] = 1 if 'missing_original_timestamp' in anomalies else 0
        features['has_future_timestamp'] = 1 if 'future_timestamp' in anomalies else 0
        features['has_identical_timestamps'] = 1 if 'identical_timestamps' in anomalies else 0
        
        # 4. Software Analysis
        software = str(exif_data.get('Software', '')).lower()
        features['software_is_empty'] = 1 if not software else 0
        features['software_is_photoshop'] = 1 if 'photoshop' in software else 0
        features['software_is_mobile'] = 1 if any(x in software for x in ['iphone', 'android', 'samsung', 'huawei']) else 0
        
        # 5. GPS Data Presence
        features['has_gps'] = 1 if any('GPS' in str(k) for k in exif_data.keys()) else 0
        
        # 6. File Metadata
        try:
            file_size = os.path.getsize(img_path)
            features['file_size_mb'] = file_size / (1024 * 1024)
        except:
            features['file_size_mb'] = 0
        
        # 7. Image Properties
        try:
            img = Image.open(img_path)
            features['image_width'] = img.width
            features['image_height'] = img.height
            features['aspect_ratio'] = img.width / img.height if img.height > 0 else 1
            features['megapixels'] = (img.width * img.height) / 1_000_000
        except:
            features.update({'image_width': 0, 'image_height': 0, 'aspect_ratio': 1, 'megapixels': 0})
        
        return features, exif_data

def main():
    print("🔒 CYBERSECURITY METADATA FORENSICS SYSTEM")
    print("="*80)
    print("Digital Forensics for AI Image Detection")
    print("="*80)
    
    # Initialize forensics analyzer
    forensics = MetadataForensics()
    
    # Load dataset
    BASE_PATH = r"C:\Users\Vatase Radu\Downloads\datetrainingFACE"
    REAL_PATH = os.path.join(BASE_PATH, "training_real") 
    FAKE_PATH = os.path.join(BASE_PATH, "training_fake")
    
    if not os.path.exists(REAL_PATH):
        print("❌ Dataset not found")
        return
    
    print(f"\n📂 Dataset: {BASE_PATH}")
    
    real_files = glob.glob(os.path.join(REAL_PATH, "*.*"))
    fake_files = glob.glob(os.path.join(FAKE_PATH, "*.*"))
    
    print(f"   Found: {len(real_files)} REAL, {len(fake_files)} FAKE")
    
    # Sample
    SAMPLE_SIZE = 1500
    if len(real_files) > SAMPLE_SIZE:
        random.seed(42)
        random.shuffle(real_files)
        random.shuffle(fake_files)
        real_files = real_files[:SAMPLE_SIZE]
        fake_files = fake_files[:SAMPLE_SIZE]
    
    all_files = real_files + fake_files
    labels = [0] * len(real_files) + [1] * len(fake_files)
    
    # Extract metadata features
    print(f"\n🔍 Analyzing metadata from {len(all_files)} images...")
    
    features_list = []
    valid_labels = []
    ai_detections = []
    metadata_samples = []
    
    for file, label in tqdm(zip(all_files, labels), total=len(all_files)):
        try:
            features, exif_data = forensics.extract_features(file)
            features_list.append(features)
            valid_labels.append(label)
            
            # Track AI detections
            if features['has_ai_signature']:
                ai_detections.append((file, label, features['ai_confidence']))
            
            # Sample metadata for report
            if len(metadata_samples) < 5:
                metadata_samples.append((os.path.basename(file), label, exif_data))
                
        except Exception as e:
            continue
    
    print(f"\n✅ Analyzed {len(features_list)} images")
    print(f"   AI Signatures Found: {len(ai_detections)}")
    
    # Convert to arrays
    feature_names = list(features_list[0].keys())
    X = np.array([[f[name] for name in feature_names] for f in features_list])
    y = np.array(valid_labels)
    
    print(f"\n📊 Features Extracted: {len(feature_names)}")
    print(f"   Samples: REAL={np.sum(y==0)}, FAKE={np.sum(y==1)}")
    
    # Metadata Statistics
    print("\n" + "="*80)
    print("📈 METADATA STATISTICS")
    print("="*80)
    
    real_features = [f for f, l in zip(features_list, valid_labels) if l == 0]
    fake_features = [f for f, l in zip(features_list, valid_labels) if l == 1]
    
    print("\n🔍 Key Metrics:")
    print(f"\nAI Signatures:")
    print(f"   REAL: {sum(f['has_ai_signature'] for f in real_features)}/{len(real_features)} ({sum(f['has_ai_signature'] for f in real_features)/len(real_features)*100:.1f}%)")
    print(f"   FAKE: {sum(f['has_ai_signature'] for f in fake_features)}/{len(fake_features)} ({sum(f['has_ai_signature'] for f in fake_features)/len(fake_features)*100:.1f}%)")
    
    print(f"\nCamera Info Present:")
    print(f"   REAL: {sum(f['has_camera_info'] for f in real_features)}/{len(real_features)} ({sum(f['has_camera_info'] for f in real_features)/len(real_features)*100:.1f}%)")
    print(f"   FAKE: {sum(f['has_camera_info'] for f in fake_features)}/{len(fake_features)} ({sum(f['has_camera_info'] for f in fake_features)/len(fake_features)*100:.1f}%)")
    
    print(f"\nMetadata Completeness (avg):")
    print(f"   REAL: {np.mean([f['metadata_completeness'] for f in real_features]):.3f}")
    print(f"   FAKE: {np.mean([f['metadata_completeness'] for f in fake_features]):.3f}")
    
    print(f"\nTimestamp Anomalies (avg):")
    print(f"   REAL: {np.mean([f['timestamp_anomaly_count'] for f in real_features]):.2f}")
    print(f"   FAKE: {np.mean([f['timestamp_anomaly_count'] for f in fake_features]):.2f}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models
    print("\n" + "="*80)
    print("🤖 TRAINING ML CLASSIFIERS")
    print("="*80)
    
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=300,
            max_depth=20,
            random_state=42,
            n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            random_state=42
        )
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n📍 Training {name}...")
        model.fit(X_train_scaled, y_train)
        
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        try:
            auc = roc_auc_score(y_test, y_proba)
        except:
            auc = 0.5
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
    
    # Best model
    best_name = max(results, key=lambda k: results[k]['accuracy'])
    best_model = results[best_name]['model']
    best_acc = results[best_name]['accuracy']
    
    # Feature importance
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        print("\n" + "="*80)
        print("🔍 TOP 15 MOST IMPORTANT METADATA FEATURES")
        print("="*80)
        for i in range(min(15, len(feature_names))):
            idx = indices[i]
            print(f"{i+1:2d}. {feature_names[idx]:30s} : {importances[idx]:.4f}")
    
    # Final Results
    print("\n" + "="*80)
    print("📈 FINAL RESULTS - METADATA FORENSICS")
    print("="*80)
    
    for name, res in results.items():
        print(f"\n{'='*80}")
        print(f"🏆 {name}")
        print(f"{'='*80}")
        print(f"Accuracy:  {res['accuracy']*100:.2f}%")
        print(f"ROC AUC:   {res['auc']:.4f}")
        cm = res['cm']
        print(f"\nConfusion Matrix:")
        print(f"   TN={cm[0,0]:4d} | FP={cm[0,1]:4d}")
        print(f"   FN={cm[1,0]:4d} | TP={cm[1,1]:4d}")
        print(f"\nClassification Report:")
        print(classification_report(y_test, res['y_pred'], target_names=['REAL', 'FAKE']))
    
    # Save model
    with open('metadata_forensics_model.pkl', 'wb') as f:
        pickle.dump({
            'model': best_model,
            'scaler': scaler,
            'feature_names': feature_names,
            'forensics': forensics
        }, f)
    print(f"\n💾 Model saved: metadata_forensics_model.pkl")
    
    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Model comparison
    names = list(results.keys())
    accs = [results[n]['accuracy']*100 for n in names]
    colors = ['gold' if n == best_name else 'steelblue' for n in names]
    
    bars = axes[0, 0].bar(names, accs, color=colors, edgecolor='black', linewidth=2)
    axes[0, 0].axhline(y=90, color='red', linestyle='--', linewidth=2, label='Target 90%')
    axes[0, 0].set_ylabel('Accuracy (%)', fontweight='bold')
    axes[0, 0].set_title('Metadata Forensics - Model Comparison', fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(axis='y', alpha=0.3)
    for bar, acc in zip(bars, accs):
        height = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width()/2., height,
                       f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    # Confusion Matrix
    best_cm = results[best_name]['cm']
    cm_norm = best_cm.astype('float') / best_cm.sum(axis=1)[:, np.newaxis]
    im = axes[0, 1].imshow(cm_norm, cmap='Blues', aspect='auto')
    axes[0, 1].set_xticks([0, 1])
    axes[0, 1].set_yticks([0, 1])
    axes[0, 1].set_xticklabels(['REAL', 'FAKE'])
    axes[0, 1].set_yticklabels(['REAL', 'FAKE'])
    axes[0, 1].set_title(f'Best Model: {best_name}\\nAccuracy: {best_acc*100:.2f}%', fontweight='bold')
    for i in range(2):
        for j in range(2):
            axes[0, 1].text(j, i, f'{cm_norm[i, j]:.2f}\\n({best_cm[i, j]})',
                           ha="center", va="center", fontweight='bold', fontsize=12)
    
    # Feature Importance
    if hasattr(best_model, 'feature_importances_'):
        top_n = 15
        top_indices = indices[:top_n]
        axes[1, 0].barh(range(top_n), importances[top_indices], color='steelblue')
        axes[1, 0].set_yticks(range(top_n))
        axes[1, 0].set_yticklabels([feature_names[i] for i in top_indices], fontsize=9)
        axes[1, 0].set_xlabel('Importance', fontweight='bold')
        axes[1, 0].set_title(f'Top 15 Metadata Features', fontweight='bold')
        axes[1, 0].invert_yaxis()
    
    # Metadata Completeness Distribution
    real_completeness = [f['metadata_completeness'] for f in real_features]
    fake_completeness = [f['metadata_completeness'] for f in fake_features]
    axes[1, 1].hist(real_completeness, bins=20, alpha=0.6, label='REAL', color='green')
    axes[1, 1].hist(fake_completeness, bins=20, alpha=0.6, label='FAKE', color='red')
    axes[1, 1].set_xlabel('Metadata Completeness Score', fontweight='bold')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Metadata Completeness Distribution', fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('CYBERSECURITY METADATA FORENSICS - AI IMAGE DETECTION', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('metadata_forensics_results.png', dpi=150, bbox_inches='tight')
    print("📊 Results saved: metadata_forensics_results.png")
    
    # Save detailed report
    with open('metadata_forensics_report.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Feature', 'Value'])
        writer.writerow(['Best Model', best_name])
        writer.writerow(['Accuracy', f'{best_acc*100:.2f}%'])
        writer.writerow(['AUC', f'{results[best_name]["auc"]:.4f}'])
        writer.writerow([''])
        writer.writerow(['Top Features'])
        for i in range(min(10, len(feature_names))):
            idx = indices[i]
            writer.writerow([feature_names[idx], f'{importances[idx]:.4f}'])
    
    print("📄 Report saved: metadata_forensics_report.csv")
    
    # Final Summary
    print("\n" + "="*80)
    print("✨ CYBERSECURITY FORENSICS SUMMARY")
    print("="*80)
    print(f"🎯 Best Accuracy: {best_acc*100:.2f}% ({best_name})")
    print(f"🔒 Features Used: {len(feature_names)} metadata attributes")
    print(f"🔍 AI Signatures Detected: {len(ai_detections)} images")
    print(f"\n💡 Key Findings:")
    print(f"   • Metadata completeness is a strong indicator")
    print(f"   • AI-generated images often lack camera EXIF data")
    print(f"   • Timestamp anomalies can reveal synthetic images")
    print(f"   • Software signatures provide direct AI tool identification")
    print("="*80)
    
    plt.show()

if __name__ == "__main__":
    main()
