# Pix2Pix GAN - Image-to-Image Translation using TensorFlow/Keras

A modern and complete implementation of the **Pix2Pix** conditional GAN for image-to-image translation tasks such as black-and-white to color, sketches to photos, or facades to buildings.

#Example


![image](https://github.com/bhautik2005/PRODIGY_GA_04/blob/a90a5408a55c6c909f39531cb6942bb1be56d6bb/image.jpg)
---

## 🚀 Key Features

* ✅ Full training pipeline with configurable parameters
* 🔁 Inference script for both single and batch image generation
* 🧰 Data preparation tools (split & concat modes)
* 📊 Visualizations of training progress and sample generations
* ⚙️ Highly customizable via `config.py`

---

## 📁 Project Structure

```bash
Pix2Pix_2_project/
├── train.py              # Main training script
├── inference.py          # Generate output images
├── main.py               # Model architecture and core logic
├── prepare_data.py       # Data preparation utilities
├── config.py             # Configuration file
├── utils.py              # Helper functions
├── requirements.txt      # List of dependencies
├── setup.py              # Optional: for packaging
└── pix2pix_output/       # Generated during training
    ├── generated_images/ # Sample outputs
    ├── saved_models/     # Generator and discriminator checkpoints
    └── logs/             # Training logs
```

---

## 🛠 Installation

### 1. Clone the repository

```bash
git clone <repo-url>
cd Pix2Pix_2_project
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Verify TensorFlow installation

```bash
python -c "import tensorflow as tf; print(tf.__version__)"
```

---

## 🧾 Data Preparation

Pix2Pix expects paired images (input | target) side by side. Prepare your dataset using:

### Option 1: Split Mode (for pre-concatenated images)

```bash
python prepare_data.py --mode split \
  --input_dir /path/to/pix2pix/images \
  --output_dir /path/to/prepared/data
```

### Option 2: Concat Mode (merge input & target images)

```bash
python prepare_data.py --mode concat \
  --input_dir /path/to/input/images \
  --target_dir /path/to/target/images \
  --output_dir /path/to/pix2pix/format
```

---

## 🏋️ Training

### Quick Start

```bash
python train.py --data_path /path/to/data --epochs 200
```

### Full Example

```bash
python train.py \
  --data_path /path/to/data \
  --epochs 200 \
  --output_dir pix2pix_output \
  --learning_rate 0.0002
```

### Key Parameters

* `--data_path` : Directory of training data (required)
* `--epochs` : Number of training epochs (default: 200)
* `--learning_rate` : Learning rate (default: 0.0002)
* `--output_dir` : Directory to save outputs

---

## 🖼️ Inference

### Single Image

```bash
python inference.py \
  --model_path pix2pix_output/saved_models/generator_epoch_200.h5 \
  --input_image /path/to/input.jpg \
  --output_path /path/to/output.png
```

### Batch Mode

```bash
python inference.py \
  --model_path pix2pix_output/saved_models/generator_epoch_200.h5 \
  --input_folder /input/images/ \
  --output_folder /output/images/
```

### Side-by-Side Comparison

```bash
python inference.py \
  --model_path pix2pix_output/saved_models/generator_epoch_200.h5 \
  --input_image /path/to/input.jpg \
  --output_path /path/to/output_comparison.png \
  --comparison
```

---

## ⚙️ Configuration Overview (`config.py`)

### Training Settings

* Batch size, buffer size, image dimensions
* Learning rate, epochs, loss weights
* Output logging frequency

### Model Architecture

* U-Net generator with skip connections
* PatchGAN discriminator
* Dropout, filters, and normalization options

### Environment

* GPU memory growth
* Mixed precision training (optional)
* Reproducibility (seed control)

---

## 🧠 Architecture Overview

### Generator: U-Net

* 8 Downsampling blocks (Conv2D + BatchNorm + LeakyReLU)
* 7 Upsampling blocks (Conv2DTranspose + BatchNorm + ReLU)
* Skip connections between corresponding encoder-decoder layers

### Discriminator: PatchGAN

* Concatenates input and target image
* 4-layer convolutional net
* Outputs 70x70 real/fake patches

---

## 📉 Loss Functions

### Generator Loss

```python
L_G = L_GAN + lambda * L_L1
```

* **L\_GAN**: Adversarial loss (BCE)
* **L\_L1**: Pixel-wise L1 loss
* **lambda**: Weighting factor (default: 100)

### Discriminator Loss

```python
L_D = L_real + L_fake
```

* Distinguishes real and fake pairs

---

## 📦 Output Directory Structure

```bash
pix2pix_output/
├── generated_images/
│   ├── image_at_epoch_001.png
│   └── ...
├── saved_models/
│   ├── generator_epoch_200.h5
│   ├── discriminator_epoch_200.h5
├── logs/
│   └── training_log_YYYYMMDD_HHMMSS.txt
```

---

## 🧩 Troubleshooting & Tips

### Common Issues

* **Out of memory**: Reduce batch size or image size
* **Training too slow**: Enable GPU or use mixed precision
* **Poor output quality**: Ensure dataset is properly paired
* **Import errors**: Recheck your `requirements.txt`

### Performance Tips

* 🧠 Use a modern GPU with Tensor Cores
* 🏎️ Enable mixed precision training
* 📥 Use `tf.data` prefetch for performance boost

---

## 🔎 Example Workflow

### Train on Facades Dataset

```bash
wget https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/facades.tar.gz

tar -xzf facades.tar.gz

python prepare_data.py --mode split \
  --input_dir facades/train \
  --output_dir prepared_data

python train.py --data_path prepared_data --epochs 200
```

### Generate a Facade

```bash
python inference.py \
  --model_path pix2pix_output/saved_models/generator_epoch_200.h5 \
  --input_image test_facade.jpg \
  --output_path generated_facade.png \
  --comparison
```

---

## 📜 License

This project is licensed under the **MIT License**. See the LICENSE file for details.

---

## 🙏 Acknowledgments

* [Pix2Pix Paper](https://arxiv.org/abs/1611.07004) by Isola et al.
* Berkeley AI Research (BAIR)
* TensorFlow/Keras Community
