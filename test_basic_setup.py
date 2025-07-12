#!/usr/bin/env python3
"""
Basic test script to verify pix2pix project setup (without TensorFlow)
Run this to check if basic components are working correctly
"""

import os
import sys
import numpy as np
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
    
    return True

def test_tensorflow():
    """Test TensorFlow availability"""
    print("\nTesting TensorFlow availability...")
    
    try:
        import tensorflow as tf  # type: ignore
        print(f"✓ TensorFlow imported successfully (version: {tf.__version__})")
        
        # Test GPU availability
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"✓ GPU found: {len(gpus)} device(s)")
            for i, gpu in enumerate(gpus):
                print(f"  GPU {i}: {gpu.name}")
        else:
            print("⚠ No GPU found - training will use CPU (slower)")
        
        return True
    except ImportError as e:
        print(f"✗ TensorFlow not available: {e}")
        print("  Please install TensorFlow using one of the methods in SETUP_GUIDE.md")
        return False

def test_data_processing():
    """Test data processing functions"""
    print("\nTesting data processing...")
    
    try:
        # Create a dummy image
        dummy_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        dummy_path = "test_image.png"
        Image.fromarray(dummy_image).save(dummy_path)
        
        # Test basic image operations
        img = Image.open(dummy_path)
        img_resized = img.resize((256, 256))
        img_array = np.array(img_resized)
        
        print("✓ Basic image processing works")
        
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

def test_data_preparation():
    """Test data preparation utilities"""
    print("\nTesting data preparation utilities...")
    
    try:
        from prepare_data import split_pix2pix_image
        
        # Create a dummy concatenated image
        dummy_input = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        dummy_target = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        
        # Concatenate horizontally
        concatenated = np.concatenate([dummy_input, dummy_target], axis=1)
        concatenated_img = Image.fromarray(concatenated)
        concatenated_path = "test_concatenated.png"
        concatenated_img.save(concatenated_path)
        
        # Test splitting
        output_dir = "test_output"
        os.makedirs(output_dir, exist_ok=True)
        
        input_path, target_path = split_pix2pix_image(concatenated_path, output_dir)
        
        print("✓ Data preparation utilities work")
        
        # Clean up
        os.remove(concatenated_path)
        import shutil
        shutil.rmtree(output_dir)
        
    except Exception as e:
        print(f"✗ Data preparation failed: {e}")
        return False
    
    return True

def main():
    """Run all tests"""
    print("Pix2Pix Project Basic Setup Test")
    print("=" * 40)
    
    tests = [
        test_imports,
        test_tensorflow,
        test_data_processing,
        test_config,
        test_output_directories,
        test_data_preparation
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
    elif passed >= total - 1:  # Allow TensorFlow to be missing
        print("⚠ Basic setup is working, but TensorFlow is not installed.")
        print("Please install TensorFlow to use the full functionality:")
        print("1. Follow the instructions in SETUP_GUIDE.md")
        print("2. Or use Google Colab for cloud-based training")
    else:
        print("✗ Some tests failed. Please check the errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 