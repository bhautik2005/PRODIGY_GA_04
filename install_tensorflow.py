#!/usr/bin/env python3
"""
TensorFlow Installation Helper Script
This script helps you install TensorFlow for the pix2pix project
"""

import sys
import subprocess
import platform

def check_python_version():
    """Check if Python version is compatible with TensorFlow"""
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor >= 8 and version.minor <= 11:
        print("✓ Python version is compatible with TensorFlow")
        return True
    elif version.major == 3 and version.minor == 12:
        print("⚠ Python 3.12 has limited TensorFlow support")
        return True
    elif version.major == 3 and version.minor >= 13:
        print("❌ Python 3.13+ is not yet supported by TensorFlow")
        return False
    else:
        print("❌ Python version not supported")
        return False

def try_install_tensorflow():
    """Try different TensorFlow installation methods"""
    print("\nAttempting TensorFlow installation...")
    
    # Method 1: Try standard TensorFlow
    print("\n1. Trying standard TensorFlow...")
    try:
        result = subprocess.run([sys.executable, "-m", "pip", "install", "tensorflow"], 
                              capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            print("✓ TensorFlow installed successfully!")
            return True
        else:
            print(f"✗ Standard TensorFlow failed: {result.stderr}")
    except Exception as e:
        print(f"✗ Standard TensorFlow failed: {e}")
    
    # Method 2: Try TensorFlow CPU
    print("\n2. Trying TensorFlow CPU...")
    try:
        result = subprocess.run([sys.executable, "-m", "pip", "install", "tensorflow-cpu"], 
                              capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            print("✓ TensorFlow CPU installed successfully!")
            return True
        else:
            print(f"✗ TensorFlow CPU failed: {result.stderr}")
    except Exception as e:
        print(f"✗ TensorFlow CPU failed: {e}")
    
    # Method 3: Try specific version
    print("\n3. Trying TensorFlow 2.10.0...")
    try:
        result = subprocess.run([sys.executable, "-m", "pip", "install", "tensorflow==2.10.0"], 
                              capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            print("✓ TensorFlow 2.10.0 installed successfully!")
            return True
        else:
            print(f"✗ TensorFlow 2.10.0 failed: {result.stderr}")
    except Exception as e:
        print(f"✗ TensorFlow 2.10.0 failed: {e}")
    
    return False

def test_tensorflow():
    """Test if TensorFlow works correctly"""
    print("\nTesting TensorFlow installation...")
    try:
        import tensorflow as tf  # type: ignore
        print(f"✓ TensorFlow imported successfully (version: {tf.__version__})")

        # Test basic functionality
        a = tf.constant([1, 2, 3], dtype=tf.int32)
        b = tf.constant([4, 5, 6])
        c = a + b
        print(f"✓ Basic TensorFlow operations work: {c.numpy()}")
        
        return True
    except Exception as e:
        print(f"✗ TensorFlow test failed: {e}")
        return False

def main():
    """Main installation helper"""
    print("TensorFlow Installation Helper")
    print("=" * 40)
    
    # Check Python version
    if not check_python_version():
        print("\n❌ Your Python version is not compatible with TensorFlow.")
        print("\nSolutions:")
        print("1. Install Python 3.11: https://www.python.org/downloads/")
        print("2. Use Google Colab for cloud-based training")
        print("3. Use Docker with Python 3.11")
        print("4. Use conda to create a Python 3.11 environment")
        return 1
    
    # Try to install TensorFlow
    if try_install_tensorflow():
        # Test the installation
        if test_tensorflow():
            print("\n🎉 TensorFlow installation successful!")
            print("\nYou can now:")
            print("1. Run: python test_setup.py")
            print("2. Train models: python train.py --data_path your_data --epochs 200")
            print("3. Generate images: python inference.py --model_path model.h5 --input_image test.jpg")
            return 0
        else:
            print("\n❌ TensorFlow installation failed verification.")
            return 1
    else:
        print("\n❌ All TensorFlow installation methods failed.")
        print("\nAlternative solutions:")
        print("1. Use Google Colab (recommended for Python 3.13)")
        print("2. Install Python 3.11 and create a virtual environment")
        print("3. Use Docker: docker run -it python:3.11-slim bash")
        print("4. Use conda: conda create -n pix2pix python=3.11")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 