"""
Script pentru înlocuirea path-urilor hardcoded cu placeholder-uri generice
Rulează acest script înainte de commit pentru a elimina date personale
"""

import os
import re

# Path-uri de înlocuit
PERSONAL_PATHS = [
    r'C:\\Users\\Vatase Radu\\Downloads\\datetrainingFACE',
    r'C:/Users/Vatase Radu/Downloads/datetrainingFACE',
    r'"C:\Users\Vatase Radu\Downloads\datetrainingFACE"',
    r"'C:\Users\Vatase Radu\Downloads\datetrainingFACE'",
]

GENERIC_PATH = r'# TODO: Set your local dataset path here (see SETUP_LOCAL.md)\nDATASET_PATH = r"/path/to/dataset"  # Change this!'

files_to_clean = [
    'train_simple_face.py',
    'train_enhanced_face.py',
    'train_advanced_face.py',
    'batch_test.py',
    'test_real_images.py',
    'test_fake_images.py',
    'test_app_logic.py',
    'calibrate_face_dataset.py',
    'retrain_model_face800.py',
    'retrain_combined_optimized.py',
    'ultimate_hybrid.py',
    'optimized_detection.py',
    'advanced_hybrid_analysis.py',
    'cnn_detection.py',
    'final_integrated_system.py'
]

def clean_file(filepath):
    """Înlocuiește path-uri personale cu placeholder-uri"""
    if not os.path.exists(filepath):
        print(f"⏭️  Skip: {filepath} (nu există)")
        return
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # Înlocuiește toate variantele de path personal
    for personal_path in PERSONAL_PATHS:
        content = content.replace(personal_path, '/path/to/dataset')
    
    # Pattern pentru linii complete cu FAKE_PATH/REAL_PATH
    content = re.sub(
        r'FAKE_PATH = r?"[^"]*datetrainingFACE[^"]*"',
        'FAKE_PATH = r"/path/to/dataset/training_fake"  # TODO: Set your path (see SETUP_LOCAL.md)',
        content
    )
    
    content = re.sub(
        r'REAL_PATH = r?"[^"]*datetrainingFACE[^"]*"',
        'REAL_PATH = r"/path/to/dataset/training_real"  # TODO: Set your path (see SETUP_LOCAL.md)',
        content
    )
    
    content = re.sub(
        r'DATASET_PATH = r?"[^"]*datetrainingFACE[^"]*"',
        'DATASET_PATH = r"/path/to/dataset"  # TODO: Set your path (see SETUP_LOCAL.md)',
        content
    )
    
    content = re.sub(
        r'BASE_PATH = r?"[^"]*datetrainingFACE[^"]*"',
        'BASE_PATH = r"/path/to/dataset"  # TODO: Set your path (see SETUP_LOCAL.md)',
        content
    )
    
    content = re.sub(
        r'FACE_PATH = "[^"]*datetrainingFACE[^"]*"',
        'FACE_PATH = "/path/to/dataset"  # TODO: Set your path (see SETUP_LOCAL.md)',
        content
    )
    
    content = re.sub(
        r'REAL_IMAGE = r?"[^"]*datetrainingFACE[^"]*"',
        'REAL_IMAGE = r"/path/to/dataset/training_real/sample.jpg"  # TODO: Set your path',
        content
    )
    
    if content != original_content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✅ Cleaned: {filepath}")
        return True
    else:
        print(f"⏭️  No changes: {filepath}")
        return False

def main():
    print("=" * 60)
    print("Curățare Path-uri Personale din Cod")
    print("=" * 60)
    print()
    
    cleaned_count = 0
    
    for filename in files_to_clean:
        if clean_file(filename):
            cleaned_count += 1
    
    print()
    print("=" * 60)
    print(f"✅ Finished! {cleaned_count} files cleaned")
    print("=" * 60)
    print()
    print("⚠️  IMPORTANT: Verifică fișierele modificate înainte de commit!")
    print("💡 TIP: Creează un .env sau config.py local pentru path-uri")

if __name__ == "__main__":
    main()
