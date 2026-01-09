"""
DEEP LEARNING CNN FOR AI IMAGE DETECTION
=========================================
Target: 90%+ Accuracy using Transfer Learning

Approach:
- EfficientNetB0 (pre-trained on ImageNet)
- Fine-tuning top layers
- Data Augmentation
- Mixed traditional features + CNN
"""

import os
import glob
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, accuracy_score
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# Configuration
IMG_SIZE = 224  # EfficientNet input size
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001

def load_and_preprocess_image(img_path, target_size=(IMG_SIZE, IMG_SIZE)):
    """Load and preprocess image"""
    try:
        img = Image.open(img_path).convert('RGB')
        img = img.resize(target_size, Image.Resampling.LANCZOS)
        img_array = np.array(img) / 255.0  # Normalize to [0, 1]
        return img_array
    except:
        return None

def create_cnn_model():
    """Create CNN model with EfficientNetB0 backbone"""
    
    # Load pre-trained EfficientNetB0
    base_model = EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    
    # Freeze base model initially
    base_model.trainable = False
    
    # Build model
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    
    # Data augmentation layer
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.05)(x)
    x = layers.RandomZoom(0.05)(x)
    
    # Base model
    x = base_model(x, training=False)
    
    # Add custom layers
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = keras.Model(inputs, outputs)
    
    return model, base_model

def main():
    print("🎯 DEEP LEARNING CNN FOR AI DETECTION")
    print("="*80)
    print("Using: EfficientNetB0 + Transfer Learning")
    print("="*80)
    
    # Check GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✅ GPU Available: {len(gpus)} device(s)")
        for gpu in gpus:
            print(f"   {gpu}")
    else:
        print("⚠️  No GPU found - using CPU (will be slower)")
    
    # === LOAD DATA ===
    BASE_PATH = r/path/to/dataset
    REAL_PATH = os.path.join(BASE_PATH, "training_real") 
    FAKE_PATH = os.path.join(BASE_PATH, "training_fake")
    
    print(f"\n📂 Loading dataset from: {BASE_PATH}")
    
    if not os.path.exists(REAL_PATH):
        print(f"❌ Error: {REAL_PATH} not found")
        return
    
    # Get file paths
    real_files = glob.glob(os.path.join(REAL_PATH, "*.*"))
    fake_files = glob.glob(os.path.join(FAKE_PATH, "*.*"))
    
    print(f"   Found: {len(real_files)} REAL, {len(fake_files)} FAKE")
    
    # Sample data (use more for better results)
    SAMPLE_SIZE = 1500
    if len(real_files) > SAMPLE_SIZE:
        print(f"⚡ Using {SAMPLE_SIZE} images per class")
        random.shuffle(real_files)
        random.shuffle(fake_files)
        real_files = real_files[:SAMPLE_SIZE]
        fake_files = fake_files[:SAMPLE_SIZE]
    
    all_files = real_files + fake_files
    labels = [0] * len(real_files) + [1] * len(fake_files)  # 0=REAL, 1=FAKE
    
    # Split data
    train_files, test_files, y_train, y_test = train_test_split(
        all_files, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    train_files, val_files, y_train, y_val = train_test_split(
        train_files, y_train, test_size=0.15, random_state=42, stratify=y_train
    )
    
    print(f"\n📊 Dataset Split:")
    print(f"   Training:   {len(train_files)} images")
    print(f"   Validation: {len(val_files)} images")
    print(f"   Test:       {len(test_files)} images")
    
    # Load images
    print("\n🔄 Loading images...")
    
    def load_batch(files, labels):
        images = []
        valid_labels = []
        for file, label in tqdm(zip(files, labels), total=len(files)):
            img = load_and_preprocess_image(file)
            if img is not None:
                images.append(img)
                valid_labels.append(label)
        return np.array(images), np.array(valid_labels)
    
    print("   Loading training set...")
    X_train, y_train = load_batch(train_files, y_train)
    print("   Loading validation set...")
    X_val, y_val = load_batch(val_files, y_val)
    print("   Loading test set...")
    X_test, y_test = load_batch(test_files, y_test)
    
    print(f"\n✅ Loaded:")
    print(f"   Training: {X_train.shape}")
    print(f"   Validation: {X_val.shape}")
    print(f"   Test: {X_test.shape}")
    
    # === BUILD MODEL ===
    print("\n🏗️  Building model...")
    model, base_model = create_cnn_model()
    
    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(name='auc')]
    )
    
    print("\n📋 Model Summary:")
    model.summary()
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        ModelCheckpoint(
            'best_cnn_model.keras',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # === TRAIN MODEL (Phase 1: Frozen base) ===
    print("\n" + "="*80)
    print("🚀 PHASE 1: Training with frozen base model")
    print("="*80)
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=20,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )
    
    # === FINE-TUNING (Phase 2: Unfreeze top layers) ===
    print("\n" + "="*80)
    print("🔥 PHASE 2: Fine-tuning (unfreezing top layers)")
    print("="*80)
    
    # Unfreeze top 20 layers
    base_model.trainable = True
    for layer in base_model.layers[:-20]:
        layer.trainable = False
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE / 10),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(name='auc')]
    )
    
    print(f"   Trainable layers: {len([l for l in model.layers if l.trainable])}")
    
    history_fine = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
        initial_epoch=len(history.history['loss'])
    )
    
    # Combine histories
    for key in history.history.keys():
        history.history[key].extend(history_fine.history[key])
    
    # === EVALUATION ===
    print("\n" + "="*80)
    print("📈 FINAL EVALUATION")
    print("="*80)
    
    # Load best model
    model = keras.models.load_model('best_cnn_model.keras')
    
    # Predictions
    y_pred_proba = model.predict(X_test, verbose=0).flatten()
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"\n🎯 Test Accuracy: {accuracy*100:.2f}%")
    print(f"🎯 ROC AUC:       {auc:.4f}")
    
    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['REAL', 'FAKE']))
    
    print("\n📊 Confusion Matrix:")
    print(f"   TN={cm[0,0]:4d} | FP={cm[0,1]:4d}")
    print(f"   FN={cm[1,0]:4d} | TP={cm[1,1]:4d}")
    
    # === PLOTTING ===
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Training history
    axes[0, 0].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
    axes[0, 0].plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
    axes[0, 0].axhline(y=0.9, color='red', linestyle='--', label='Target 90%')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_title('Training History - Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(history.history['loss'], label='Train Loss', linewidth=2)
    axes[0, 1].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Training History - Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    axes[1, 0].plot(fpr, tpr, linewidth=3, label=f'CNN (AUC={auc:.4f})')
    axes[1, 0].plot([0, 1], [0, 1], 'k--', label='Random')
    axes[1, 0].set_xlabel('False Positive Rate')
    axes[1, 0].set_ylabel('True Positive Rate')
    axes[1, 0].set_title('ROC Curve')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Confusion Matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    im = axes[1, 1].imshow(cm_normalized, cmap='Blues', aspect='auto')
    axes[1, 1].set_xticks([0, 1])
    axes[1, 1].set_yticks([0, 1])
    axes[1, 1].set_xticklabels(['REAL', 'FAKE'])
    axes[1, 1].set_yticklabels(['REAL', 'FAKE'])
    axes[1, 1].set_xlabel('Predicted')
    axes[1, 1].set_ylabel('Actual')
    axes[1, 1].set_title(f'Confusion Matrix\\nAccuracy: {accuracy*100:.2f}%')
    
    for i in range(2):
        for j in range(2):
            text = axes[1, 1].text(j, i, f'{cm_normalized[i, j]:.2f}\\n({cm[i, j]})',
                                  ha="center", va="center", 
                                  color="white" if cm_normalized[i, j] > 0.5 else "black",
                                  fontweight='bold', fontsize=12)
    
    plt.suptitle('CNN Deep Learning - AI Image Detection', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig("cnn_results.png", dpi=150, bbox_inches='tight')
    print("\n📊 Results saved: cnn_results.png")
    
    # Final Summary
    print("\n" + "="*80)
    print("✨ FINAL SUMMARY")
    print("="*80)
    print(f"🎯 Target Accuracy: 90%")
    print(f"🏆 Achieved:        {accuracy*100:.2f}%")
    
    if accuracy >= 0.9:
        print("\n✅ ✅ ✅ TARGET REACHED! 🎉 🎉 🎉")
    else:
        gap = 90 - accuracy*100
        print(f"\n📊 Gap to target: {gap:.2f}%")
        if accuracy >= 0.85:
            print("💪 Very close! Consider:")
            print("   - Training longer (more epochs)")
            print("   - Using more data")
            print("   - Trying EfficientNetB3/B4 (larger models)")
    
    print("="*80)
    print("\n💾 Model saved: best_cnn_model.keras")
    print("📊 Results saved: cnn_results.png")
    
    plt.show()

if __name__ == "__main__":
    main()
