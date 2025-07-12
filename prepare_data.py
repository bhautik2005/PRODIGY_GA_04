#!/usr/bin/env python3
"""
Data preparation script for pix2pix GAN
This script handles the pix2pix dataset format where input and target images are concatenated
"""

import os
import argparse
import numpy as np
from PIL import Image
import glob
from pathlib import Path

def split_pix2pix_image(image_path, output_dir, split_ratio=0.5):
    """
    Split a pix2pix image (input|target) into separate input and target images
    
    Args:
        image_path: Path to the concatenated image
        output_dir: Directory to save split images
        split_ratio: Ratio of input image width (default 0.5 for equal split)
    """
    # Load image
    image = Image.open(image_path).convert('RGB')
    width, height = image.size
    
    # Calculate split point
    split_point = int(width * split_ratio)
    
    # Split image
    input_image = image.crop((0, 0, split_point, height))
    target_image = image.crop((split_point, 0, width, height))
    
    # Create output filenames
    base_name = Path(image_path).stem
    input_path = os.path.join(output_dir, f"{base_name}_input.png")
    target_path = os.path.join(output_dir, f"{base_name}_target.png")
    
    # Save split images
    input_image.save(input_path)
    target_image.save(target_path)
    
    return input_path, target_path

def prepare_dataset(input_dir, output_dir, split_ratio=0.5):
    """
    Prepare a complete dataset from pix2pix format images
    
    Args:
        input_dir: Directory containing concatenated images
        output_dir: Directory to save prepared dataset
        split_ratio: Ratio for splitting images
    """
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    image_files = []
    
    for extension in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_dir, extension)))
        image_files.extend(glob.glob(os.path.join(input_dir, extension.upper())))
    
    if not image_files:
        print(f"No image files found in {input_dir}")
        return
    
    print(f"Found {len(image_files)} images to process")
    
    # Process each image
    for i, image_path in enumerate(image_files):
        try:
            split_pix2pix_image(image_path, output_dir, split_ratio)
            print(f"Processed ({i+1}/{len(image_files)}): {os.path.basename(image_path)}")
        except Exception as e:
            print(f"Error processing {os.path.basename(image_path)}: {str(e)}")
    
    print(f"Dataset preparation completed! Results saved to: {output_dir}")

def create_pix2pix_format(input_dir, target_dir, output_dir):
    """
    Create pix2pix format images from separate input and target images
    
    Args:
        input_dir: Directory containing input images
        target_dir: Directory containing target images
        output_dir: Directory to save concatenated images
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get input and target files
    input_files = glob.glob(os.path.join(input_dir, "*.jpg"))
    input_files.extend(glob.glob(os.path.join(input_dir, "*.png")))
    input_files.extend(glob.glob(os.path.join(input_dir, "*.jpeg")))
    
    target_files = glob.glob(os.path.join(target_dir, "*.jpg"))
    target_files.extend(glob.glob(os.path.join(target_dir, "*.png")))
    target_files.extend(glob.glob(os.path.join(target_dir, "*.jpeg")))
    
    if not input_files or not target_files:
        print("No image files found in input or target directories")
        return
    
    print(f"Found {len(input_files)} input images and {len(target_files)} target images")
    
    # Match input and target images by filename
    input_dict = {Path(f).stem: f for f in input_files}
    target_dict = {Path(f).stem: f for f in target_files}
    
    # Find matching pairs
    matching_pairs = []
    for input_name, input_path in input_dict.items():
        if input_name in target_dict:
            matching_pairs.append((input_path, target_dict[input_name]))
    
    print(f"Found {len(matching_pairs)} matching pairs")
    
    # Create concatenated images
    for i, (input_path, target_path) in enumerate(matching_pairs):
        try:
            # Load images
            input_img = Image.open(input_path).convert('RGB')
            target_img = Image.open(target_path).convert('RGB')
            
            # Resize to same size (256x256 for pix2pix)
            input_img = input_img.resize((256, 256))
            target_img = target_img.resize((256, 256))
            
            # Concatenate horizontally
            concatenated = Image.new('RGB', (512, 256))
            concatenated.paste(input_img, (0, 0))
            concatenated.paste(target_img, (256, 0))
            
            # Save concatenated image
            base_name = Path(input_path).stem
            output_path = os.path.join(output_dir, f"{base_name}.png")
            concatenated.save(output_path)
            
            print(f"Created ({i+1}/{len(matching_pairs)}): {base_name}")
            
        except Exception as e:
            print(f"Error processing {os.path.basename(input_path)}: {str(e)}")
    
    print(f"Pix2Pix format creation completed! Results saved to: {output_dir}")

def main():
    """Main function for command line interface"""
    parser = argparse.ArgumentParser(description='Prepare data for pix2pix training')
    parser.add_argument('--mode', type=str, required=True, choices=['split', 'concat'],
                       help='Mode: split (pix2pix to separate) or concat (separate to pix2pix)')
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Input directory containing images')
    parser.add_argument('--target_dir', type=str,
                       help='Target directory (only for concat mode)')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for processed images')
    parser.add_argument('--split_ratio', type=float, default=0.5,
                       help='Split ratio for input image width (default: 0.5)')
    
    args = parser.parse_args()
    
    if args.mode == 'split':
        # Split pix2pix format images into separate input and target
        prepare_dataset(args.input_dir, args.output_dir, args.split_ratio)
    
    elif args.mode == 'concat':
        # Create pix2pix format from separate input and target images
        if not args.target_dir:
            print("Error: --target_dir is required for concat mode")
            return
        create_pix2pix_format(args.input_dir, args.target_dir, args.output_dir)

if __name__ == "__main__":
    main() 