#!/usr/bin/env python3
"""
Test script to verify pix2pix project setup
Run this to check if all components are working correctly
"""

import os
import sys
import numpy as np
import tensorflow as tf  # type: ignore
from PIL import Image

def test_imports():
    """Test if all required modules can be imported"""
    print("Testing imports...")
    
    try:
        import matplotlib.pyplot as plt
        print("✓ matplotlib imported successfully")
    except ImportError as e:
        print(f"✗ matplotlib import failed: {e}")
        return False
    
    try:
        from PIL import Image
        print("✓ PIL/Pillow imported successfully")
    except ImportError as e:
        print(f"✗ PIL import failed: {e}")
        return False
    
    try:
        import numpy as np
        print("✓ numpy imported successfully")
    except ImportError as e:
        print(f"✗ numpy import failed: {e}")
        return False
    
    try:
        import tensorflow as tf  # type: ignore
        print(f"✓ tensorflow imported successfully (version: {tf.__version__})")
    except ImportError as e:
        print(f"✗ tensorflow import failed: {e}")
        return False
    
    return True

def test_gpu():
    """Test GPU availability"""
    print("\nTesting GPU availability...")
    
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✓ GPU found: {len(gpus)} device(s)")
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}: {gpu.name}")
        return True
    else:
        print("⚠ No GPU found - training will use CPU (slower)")
        return True

def test_model_creation():
    """Test if models can be created"""
    print("\nTesting model creation...")
    
    try:
        from main import create_generator
        generator = create_generator()
        print("✓ Generator model created successfully")
        print(f"  Generator parameters: {generator.count_params():,}")
    except Exception as e:
        print(f"✗ Generator creation failed: {e}")
        return False
    
    try:
        from train import create_discriminator
        discriminator = create_discriminator()
        print("✓ Discriminator model created successfully")
        print(f"  Discriminator parameters: {discriminator.count_params():,}")
    except Exception as e:
        print(f"✗ Discriminator creation failed: {e}")
        return False
    
    return True

def test_data_processing():
    """Test data processing functions"""
    print("\nTesting data processing...")
    
    try:
        from main import load_and_preprocess_image, postprocess_image
        
        # Create a dummy image
        dummy_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        dummy_path = "test_image.png"
        Image.fromarray(dummy_image).save(dummy_path)
        
        # Test preprocessing
        processed = load_and_preprocess_image(dummy_path)
        print("✓ Image preprocessing works")
        
        # Test postprocessing
        postprocessed = postprocess_image(processed)
        print("✓ Image postprocessing works")
        
        # Clean up
        os.remove(dummy_path)
        
    except Exception as e:
        print(f"✗ Data processing failed: {e}")
        return False
    
    return True

def test_config():
    """Test configuration loading"""
    print("\nTesting configuration...")
    
    try:
        from config import TrainingConfig, ModelConfig
        
        # Test training config
        train_config = TrainingConfig()
        print("✓ Training configuration loaded")
        print(f"  Image size: {train_config.IMG_HEIGHT}x{train_config.IMG_WIDTH}")
        print(f"  Batch size: {train_config.BATCH_SIZE}")
        print(f"  Epochs: {train_config.EPOCHS}")
        
        # Test model config
        model_config = ModelConfig()
        print("✓ Model configuration loaded")
        
    except Exception as e:
        print(f"✗ Configuration loading failed: {e}")
        return False
    
    return True

def test_output_directories():
    """Test output directory creation"""
    print("\nTesting output directories...")
    
    try:
        from config import TrainingConfig
        
        output_dir = TrainingConfig.OUTPUT_DIR
        images_dir = os.path.join(output_dir, TrainingConfig.IMAGES_DIR)
        models_dir = os.path.join(output_dir, TrainingConfig.MODELS_DIR)
        logs_dir = os.path.join(output_dir, TrainingConfig.LOGS_DIR)
        
        # Create directories
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(logs_dir, exist_ok=True)
        
        print("✓ Output directories created successfully")
        print(f"  Images: {images_dir}")
        print(f"  Models: {models_dir}")
        print(f"  Logs: {logs_dir}")
        
    except Exception as e:
        print(f"✗ Directory creation failed: {e}")
        return False
    
    return True

def main():
    """Run all tests"""
    print("Pix2Pix Project Setup Test")
    print("=" * 40)
    
    tests = [
        test_imports,
        test_gpu,
        test_model_creation,
        test_data_processing,
        test_config,
        test_output_directories
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 40)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All tests passed! Your pix2pix setup is ready.")
        print("\nNext steps:")
        print("1. Prepare your dataset using prepare_data.py")
        print("2. Train the model using train.py")
        print("3. Generate images using inference.py")
    else:
        print("✗ Some tests failed. Please check the errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 