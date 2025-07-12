# Pix2Pix Setup Guide

## Python Version Compatibility

This project requires TensorFlow, which has specific Python version requirements:

### Supported Python Versions for TensorFlow
- **Python 3.8-3.11**: Full TensorFlow support
- **Python 3.12**: Limited TensorFlow support
- **Python 3.13**: Not yet supported by TensorFlow

## Installation Options

### Option 1: Use Python 3.11 (Recommended)

1. **Install Python 3.11**:
   - Download from [python.org](https://www.python.org/downloads/)
   - Or use conda: `conda create -n pix2pix python=3.11`

2. **Create virtual environment**:
   ```bash
   python3.11 -m venv pix2pix_env
   source pix2pix_env/bin/activate  # On Windows: pix2pix_env\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Option 2: Use Conda Environment

1. **Create conda environment**:
   ```bash
   conda create -n pix2pix python=3.11
   conda activate pix2pix
   ```

2. **Install TensorFlow**:
   ```bash
   conda install tensorflow
   ```

3. **Install other dependencies**:
   ```bash
   pip install matplotlib Pillow numpy opencv-python scikit-image tqdm seaborn
   ```

### Option 3: Use Docker

1. **Create Dockerfile**:
   ```dockerfile
   FROM python:3.11-slim
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   COPY . .
   CMD ["python", "test_setup.py"]
   ```

2. **Build and run**:
   ```bash
   docker build -t pix2pix .
   docker run pix2pix
   ```

## Current Python 3.13 Workaround

If you're using Python 3.13 and can't downgrade, you can:

1. **Install basic dependencies**:
   ```bash
   pip install numpy matplotlib Pillow opencv-python scikit-image tqdm seaborn
   ```

2. **Use alternative ML frameworks**:
   - PyTorch: `pip install torch torchvision`
   - JAX: `pip install jax jaxlib`

3. **Modify the code** to use PyTorch instead of TensorFlow (requires code changes)

## Verification

After installation, run the test script:

```bash
python test_setup.py
```

## Troubleshooting

### TensorFlow Installation Issues

1. **Check Python version**:
   ```bash
   python --version
   ```

2. **Try different TensorFlow versions**:
   ```bash
   pip install tensorflow==2.10.0
   ```

3. **Use conda instead of pip**:
   ```bash
   conda install tensorflow
   ```

### GPU Support

1. **Check GPU availability**:
   ```python
   import tensorflow as tf
   print(tf.config.list_physical_devices('GPU'))
   ```

2. **Install GPU version** (if available):
   ```bash
   pip install tensorflow-gpu
   ```

### Memory Issues

1. **Limit GPU memory growth**:
   ```python
   import tensorflow as tf
   gpus = tf.config.experimental.list_physical_devices('GPU')
   if gpus:
       tf.config.experimental.set_memory_growth(gpus[0], True)
   ```

2. **Reduce batch size** in `config.py`

## Alternative: Use Google Colab

If local installation is problematic:

1. **Upload project to Google Colab**
2. **Install dependencies in Colab**:
   ```python
   !pip install tensorflow matplotlib Pillow numpy
   ```
3. **Run training and inference in the cloud**

## Next Steps

Once TensorFlow is installed:

1. **Test the setup**: `python test_setup.py`
2. **Prepare your dataset**: Use `prepare_data.py`
3. **Train the model**: Use `train.py`
4. **Generate images**: Use `inference.py`

## Support

If you continue to have issues:

1. Check the [TensorFlow installation guide](https://www.tensorflow.org/install)
2. Consider using a different Python version
3. Use Google Colab for cloud-based training 