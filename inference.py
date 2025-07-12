#!/usr/bin/env python3
"""
Inference script for trained pix2pix model
Use this to generate images from trained model
"""

try:
    import tensorflow as tf  # type: ignore
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("⚠ TensorFlow not available. Please install TensorFlow to use this script.")
    print("   Follow the instructions in SETUP_GUIDE.md for installation options.")

import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
from pathlib import Path

# Import model architecture from main.py (only if TensorFlow is available)
if TENSORFLOW_AVAILABLE:
    from main import create_generator, load_and_preprocess_image, postprocess_image

class Pix2PixInference:
    def __init__(self, model_path):
        """Initialize inference class with trained model"""
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for inference. Please install TensorFlow first.")
        
        self.generator = create_generator()
        self.generator.load_weights(model_path)
        print(f"Model loaded from: {model_path}")
    
    def preprocess_image(self, image_path):
        """Preprocess single image for inference"""
        return load_and_preprocess_image(image_path)
    
    def generate_image(self, input_image_path, output_path=None):
        """Generate image from input"""
        # Preprocess input
        input_image = self.preprocess_image(input_image_path)
        
        # Generate output
        generated_image = self.generator(input_image, training=False)
        
        # Postprocess generated image
        generated_image = postprocess_image(generated_image.numpy())
        
        # Save or display result
        if output_path:
            # Save the generated image
            from PIL import Image
            Image.fromarray(generated_image).save(output_path)
            print(f"Generated image saved to: {output_path}")
        
        return generated_image
    
    def generate_comparison(self, input_image_path, output_path=None):
        """Generate side-by-side comparison"""
        # Load and preprocess input
        input_image = self.preprocess_image(input_image_path)
        
        # Generate output
        generated_image = self.generator(input_image, training=False)
        
        # Postprocess generated image
        generated_image = postprocess_image(generated_image.numpy())
        
        # Load original image for comparison
        from PIL import Image
        original_image = np.array(Image.open(input_image_path).convert('RGB').resize((256, 256)))
        
        # Create comparison plot
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.title('Input Image')
        plt.imshow(original_image)
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.title('Generated Image')
        plt.imshow(generated_image)
        plt.axis('off')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Comparison saved to: {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def batch_inference(self, input_folder, output_folder):
        """Run inference on all images in a folder"""
        input_folder = Path(input_folder)
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)
        
        # Supported image extensions
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        
        # Process all images in folder
        for image_path in input_folder.iterdir():
            if image_path.suffix.lower() in image_extensions:
                output_path = output_folder / f"generated_{image_path.name}"
                self.generate_image(str(image_path), str(output_path))
        
        print(f"Batch inference completed! Results saved to: {output_folder}")

def main():
    """Main function for command line interface"""
    if not TENSORFLOW_AVAILABLE:
        print("❌ TensorFlow is not installed. This script requires TensorFlow to run.")
        print("\nTo install TensorFlow:")
        print("1. Follow the instructions in SETUP_GUIDE.md")
        print("2. Or use Google Colab for cloud-based training")
        print("3. Or install Python 3.11 and create a virtual environment")
        return 1
    
    parser = argparse.ArgumentParser(description='pix2pix Inference Script')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained generator weights')
    parser.add_argument('--input_image', type=str,
                       help='Path to input image')
    parser.add_argument('--input_folder', type=str,
                       help='Path to folder containing input images')
    parser.add_argument('--output_path', type=str,
                       help='Path to save generated image')
    parser.add_argument('--output_folder', type=str,
                       help='Path to folder for saving generated images')
    parser.add_argument('--comparison', action='store_true',
                       help='Generate side-by-side comparison')
    
    args = parser.parse_args()
    
    # Initialize inference
    try:
        inference = Pix2PixInference(args.model_path)
    except Exception as e:
        print(f"❌ Error initializing inference: {e}")
        return 1
    
    if args.input_folder and args.output_folder:
        # Batch inference
        inference.batch_inference(args.input_folder, args.output_folder)
    
    elif args.input_image:
        # Single image inference
        if args.comparison:
            inference.generate_comparison(args.input_image, args.output_path)
        else:
            inference.generate_image(args.input_image, args.output_path)
    
    else:
        print("Please provide either --input_image or --input_folder")

if __name__ == "__main__":
    main()