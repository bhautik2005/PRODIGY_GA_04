#!/usr/bin/env python3
"""
Create sample pix2pix format images for testing
"""

import numpy as np
from PIL import Image
import os

def create_sample_pix2pix_image(output_path, size=(512, 256)):
    """Create a sample pix2pix format image (input|target concatenated)"""
    
    # Create input image (left side) - simple pattern
    input_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    
    # Create target image (right side) - different pattern
    target_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    
    # Concatenate horizontally (input|target)
    concatenated = np.concatenate([input_img, target_img], axis=1)
    
    # Save as image
    Image.fromarray(concatenated).save(output_path)
    print(f"Created sample image: {output_path}")

def main():
    """Create sample data directory with pix2pix format images"""
    
    # Create sample data directory
    sample_dir = "sample_data"
    os.makedirs(sample_dir, exist_ok=True)
    
    # Create 5 sample images
    for i in range(5):
        output_path = os.path.join(sample_dir, f"sample_{i+1}.png")
        create_sample_pix2pix_image(output_path)
    
    print(f"\nCreated 5 sample pix2pix images in '{sample_dir}/'")
    print("You can now test data preparation:")
    print(f"python prepare_data.py --mode split --input_dir {sample_dir} --output_dir prepared_data")

if __name__ == "__main__":
    main() 