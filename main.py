import os
import argparse
import numpy as np

try:
    import tensorflow as tf  # type: ignore
    from tensorflow import keras  # type: ignore
    from tensorflow.keras import layers  # type: ignore
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("⚠ TensorFlow not available. Some functions may not work.")

import matplotlib.pyplot as plt
from PIL import Image
import glob

def create_generator():
    """Create the U-Net generator (same as training)"""
    if not TENSORFLOW_AVAILABLE:
        raise ImportError("TensorFlow is required to create the generator model.")
    
    def downsample(filters, size, apply_batchnorm=True):
        initializer = tf.random_normal_initializer(0., 0.02)
        result = keras.Sequential()
        result.add(layers.Conv2D(filters, size, strides=2, padding='same', kernel_initializer=initializer, use_bias=False))
        
        if apply_batchnorm:
            result.add(layers.BatchNormalization())
        
        result.add(layers.LeakyReLU())
        return result

    def upsample(filters, size, apply_dropout=False):
        initializer = tf.random_normal_initializer(0., 0.02)
        result = keras.Sequential()
        result.add(layers.Conv2DTranspose(filters, size, strides=2, padding='same', kernel_initializer=initializer, use_bias=False))
        result.add(layers.BatchNormalization())
        
        if apply_dropout:
            result.add(layers.Dropout(0.5))
        
        result.add(layers.ReLU())
        return result

    inputs = layers.Input(shape=[256, 256, 3])
    
    # Downsampling layers
    down_stack = [
        downsample(64, 4, apply_batchnorm=False),
        downsample(128, 4),
        downsample(256, 4),
        downsample(512, 4),
        downsample(512, 4),
        downsample(512, 4),
        downsample(512, 4),
        downsample(512, 4),
    ]
    
    # Upsampling layers
    up_stack = [
        upsample(512, 4, apply_dropout=True),
        upsample(512, 4, apply_dropout=True),
        upsample(512, 4, apply_dropout=True),
        upsample(512, 4),
        upsample(256, 4),
        upsample(128, 4),
        upsample(64, 4),
    ]
    
    initializer = tf.random_normal_initializer(0., 0.02)
    last = layers.Conv2DTranspose(3, 4, strides=2, padding='same', kernel_initializer=initializer, activation='tanh')
    
    x = inputs
    
    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)
    
    skips = reversed(skips[:-1])
    
    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = layers.Concatenate()([x, skip])
    
    x = last(x)
    
    return keras.Model(inputs=inputs, outputs=x)

def load_and_preprocess_image(image_path, target_size=(256, 256)):
    """Load and preprocess a single image for inference"""
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Resize image
    image = image.resize(target_size)
    
    # Convert to numpy array and normalize
    image_array = np.array(image)
    image_array = (image_array / 127.5) - 1.0  # Normalize to [-1, 1]
    
    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)
    
    return image_array

def postprocess_image(image_array):
    """Convert model output back to displayable image"""
    # Remove batch dimension
    image = image_array[0]
    
    # Convert from [-1, 1] to [0, 1]
    image = (image + 1.0) / 2.0
    
    # Clip values to [0, 1]
    image = np.clip(image, 0, 1)
    
    # Convert to uint8
    image = (image * 255).astype(np.uint8)
    
    return image

def generate_single_image(generator, input_path, output_path, comparison=False):
    """Generate a single image and save it"""
    # Load and preprocess input image
    input_image = load_and_preprocess_image(input_path)
    
    # Generate image
    generated_image = generator(input_image, training=False)
    
    # Postprocess generated image
    generated_image = postprocess_image(generated_image.numpy())
    
    if comparison:
        # Create comparison image
        original = np.array(Image.open(input_path).convert('RGB').resize((256, 256)))
        
        # Create side-by-side comparison
        comparison_image = np.concatenate([original, generated_image], axis=1)
        
        # Save comparison
        Image.fromarray(comparison_image).save(output_path)
        print(f"Comparison saved to: {output_path}")
    else:
        # Save only generated image
        Image.fromarray(generated_image).save(output_path)
        print(f"Generated image saved to: {output_path}")

def batch_generate_images(generator, input_folder, output_folder):
    """Generate images for all files in input folder"""
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Get all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    image_files = []
    
    for extension in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_folder, extension)))
        image_files.extend(glob.glob(os.path.join(input_folder, extension.upper())))
    
    if not image_files:
        print(f"No image files found in {input_folder}")
        return
    
    print(f"Found {len(image_files)} images to process")
    
    for i, input_path in enumerate(image_files):
        # Generate output filename
        filename = os.path.basename(input_path)
        name, ext = os.path.splitext(filename)
        output_path = os.path.join(output_folder, f"{name}_generated{ext}")
        
        try:
            generate_single_image(generator, input_path, output_path)
            print(f"Processed ({i+1}/{len(image_files)}): {filename}")
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='pix2pix GAN Inference')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained generator model')
    parser.add_argument('--input_image', type=str, help='Path to input image')
    parser.add_argument('--input_folder', type=str, help='Path to input folder for batch processing')
    parser.add_argument('--output_path', type=str, help='Path to save generated image')
    parser.add_argument('--output_folder', type=str, help='Path to save batch generated images')
    parser.add_argument('--comparison', action='store_true', help='Generate side-by-side comparison')
    
    args = parser.parse_args()
    
    # Check if TensorFlow is available
    if not TENSORFLOW_AVAILABLE:
        print("❌ TensorFlow is not installed. This script requires TensorFlow to run.")
        print("\nTo install TensorFlow:")
        print("1. Follow the instructions in SETUP_GUIDE.md")
        print("2. Or use Google Colab for cloud-based training")
        print("3. Or install Python 3.11 and create a virtual environment")
        return 1
    
    # Validate arguments
    if not args.input_image and not args.input_folder:
        print("Error: Either --input_image or --input_folder must be specified")
        return
    
    if args.input_image and not args.output_path:
        print("Error: --output_path must be specified when using --input_image")
        return
    
    if args.input_folder and not args.output_folder:
        print("Error: --output_folder must be specified when using --input_folder")
        return
    
    # Load model
    print("Loading generator model...")
    try:
        generator = create_generator()
        generator.load_weights(args.model_path)
        print(f"Model loaded successfully from: {args.model_path}")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return
    
    # Generate images
    if args.input_image:
        # Single image generation
        generate_single_image(generator, args.input_image, args.output_path, args.comparison)
    else:
        # Batch generation
        batch_generate_images(generator, args.input_folder, args.output_folder)
    
    print("Inference completed!")

if __name__ == "__main__":
    main()