"""
Transfer Learning Models for Deepfake Detection
Optimized for 4GB VRAM - Industry Standard Architectures
Dataset: 140k Real and Fake Faces (128x128)

Supported architectures:
- Xception: Industry standard for deepfake detection (best for compression artifacts)
- EfficientNetB0: Excellent accuracy with low resource usage
- ResNet50: Classic, stable, easy to implement
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import (
    Xception,
    EfficientNetB0,
    ResNet50
)


def create_transfer_learning_model(
    architecture='xception',
    input_shape=(256, 256, 3),
    dropout_rate=0.5,
    trainable_layers=20
):
    """
    Create Transfer Learning model for deepfake detection
    
    Args:
        architecture: 'xception', 'efficientnet', or 'resnet50'
        input_shape: Input image shape (height, width, channels)
        dropout_rate: Dropout rate for regularization (prevent overfitting)
        trainable_layers: Number of layers to unfreeze for fine-tuning
    
    Returns:
        Keras model with pre-trained base
    """
    # Select base model
    if architecture.lower() == 'xception':
        # Xception: Best for deepfake detection (compression artifacts)
        base_model = Xception(
            include_top=False,  # Remove classification head
            weights='imagenet', # Use pre-trained weights
            input_shape=input_shape
        )
        
    elif architecture.lower() == 'efficientnet':
        # EfficientNetB0: Great accuracy, low resource usage
        base_model = EfficientNetB0(
            include_top=False,
            weights='imagenet',
            input_shape=input_shape
        )
        
    elif architecture.lower() == 'resnet50':
        # ResNet50: Classic, stable, reliable
        base_model = ResNet50(
            include_top=False,
            weights='imagenet',
            input_shape=input_shape
        )
        
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    
    # Freeze base model initially (recommended for transfer learning)
    base_model.trainable = False
    
    # Build complete model (simplified, as recommended)
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),  # Better than Flatten for CNNs
        layers.Dropout(dropout_rate),     # Prevent overfitting
        layers.Dense(1, activation='sigmoid')  # Binary: 0=Real, 1=Fake
    ], name=f'{architecture}_deepfake_detector')
    
    return model, base_model


def unfreeze_model(model, base_model, num_layers=20):
    """
    Unfreeze top layers of base model for fine-tuning
    
    Args:
        model: Complete model
        base_model: Base model to unfreeze
        num_layers: Number of layers to unfreeze from the top
    
    Returns:
        Model with unfrozen layers
    """
    base_model.trainable = True
    
    # Freeze all layers except the last num_layers
    for layer in base_model.layers[:-num_layers]:
        layer.trainable = False
    
    print(f"\n✅ Unfroze last {num_layers} layers of base model")
    print(f"   Trainable params: {model.count_params():,}")
    
    return model


def compile_model(model, learning_rate=0.001, stage='initial'):
    """
    Compile model with optimizer and metrics
    
    Args:
        model: Keras model to compile
        learning_rate: Initial learning rate
        stage: 'initial' (frozen base) or 'fine_tune' (unfrozen base)
    
    Returns:
        Compiled model
    """
    # Use lower learning rate for fine-tuning
    if stage == 'fine_tune':
        learning_rate = learning_rate / 10
    
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc')
        ]
    )
    
    print(f"\n✅ Model compiled for {stage} stage")
    print(f"   Learning rate: {learning_rate}")
    
    return model


def get_model_summary(model, architecture='xception'):
    """
    Print model architecture and parameter count
    """
    print("\n" + "="*60)
    print(f"TRANSFER LEARNING MODEL - {architecture.upper()}")
    print("="*60)
    model.summary()
    
    total_params = model.count_params()
    trainable_params = sum([tf.size(var).numpy() for var in model.trainable_variables])
    
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters: {total_params - trainable_params:,}")
    print(f"Estimated model size: ~{total_params * 4 / (1024**2):.1f} MB")
    print("="*60 + "\n")


if __name__ == "__main__":
    # Test all architectures
    architectures = ['xception', 'efficientnet', 'resnet50']
    
    for arch in architectures:
        print(f"\n{'='*60}")
        print(f"Testing {arch.upper()}")
        print(f"{'='*60}")
        
        try:
            model, base_model = create_transfer_learning_model(architecture=arch)
            model = compile_model(model, stage='initial')
            get_model_summary(model, arch)
            
            # Test with dummy input
            import numpy as np
            dummy_input = np.random.rand(1, 128, 128, 3).astype(np.float32)
            prediction = model.predict(dummy_input, verbose=0)
            print(f"✅ Test prediction: {prediction[0][0]:.4f}")
            print(f"   Classification: {'FAKE' if prediction[0][0] > 0.5 else 'REAL'}")
            
        except Exception as e:
            print(f"❌ Error testing {arch}: {e}")

