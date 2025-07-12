#!/usr/bin/env python3
"""
Training script for pix2pix GAN
This script handles the complete training pipeline
"""

import os
import argparse
import numpy as np
import tensorflow as tf  # type: ignore
from tensorflow import keras  # type: ignore
from tensorflow.keras import layers  # type: ignore
import matplotlib.pyplot as plt
from PIL import Image
import glob
import datetime
from pathlib import Path

# Import configurations
from config import TrainingConfig, ModelConfig, DataAugmentationConfig, EnvironmentConfig

# Import generator from main.py
from main import create_generator

def create_discriminator():
    """Create the PatchGAN discriminator"""
    initializer = tf.random_normal_initializer(0., 0.02)
    
    def downsample(filters, size, apply_batchnorm=True):
        result = keras.Sequential()
        result.add(layers.Conv2D(filters, size, strides=2, padding='same', 
                                kernel_initializer=initializer, use_bias=False))
        
        if apply_batchnorm:
            result.add(layers.BatchNormalization())
        
        result.add(layers.LeakyReLU())
        return result
    
    inputs = layers.Input(shape=[256, 256, 3])
    target = layers.Input(shape=[256, 256, 3])
    
    x = layers.Concatenate()([inputs, target])
    
    down1 = downsample(64, 4, apply_batchnorm=False)
    down2 = downsample(128, 4)
    down3 = downsample(256, 4)
    
    zero_pad1 = layers.ZeroPadding2D()
    conv = layers.Conv2D(512, 4, strides=1, kernel_initializer=initializer, use_bias=False)
    batchnorm1 = layers.BatchNormalization()
    
    zero_pad2 = layers.ZeroPadding2D()
    last = layers.Conv2D(1, 4, strides=1, kernel_initializer=initializer)
    
    x = down1(x)
    x = down2(x)
    x = down3(x)
    x = zero_pad1(x)
    x = conv(x)
    x = batchnorm1(x)
    x = layers.LeakyReLU()(x)
    x = zero_pad2(x)
    x = last(x)
    
    return keras.Model(inputs=[inputs, target], outputs=x)

def load_image(image_path):
    """Load and preprocess image"""
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=3)
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, [256, 256])
    image = (image / 127.5) - 1
    return image

def load_image_train(image_path):
    """Load and preprocess image for training with augmentation"""
    image = load_image(image_path)
    
    # Random jittering
    image = tf.image.resize(image, [286, 286], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    image = tf.image.random_crop(image, size=[256, 256, 3])
    
    # Random flip
    image = tf.image.random_flip_left_right(image)
    
    return image

def create_dataset(data_path, is_training=True):
    """Create TensorFlow dataset"""
    if is_training:
        load_fn = load_image_train
    else:
        load_fn = load_image
    
    # Get all image files
    image_files = glob.glob(os.path.join(data_path, "*.jpg"))
    image_files.extend(glob.glob(os.path.join(data_path, "*.png")))
    image_files.extend(glob.glob(os.path.join(data_path, "*.jpeg")))
    
    if not image_files:
        raise ValueError(f"No image files found in {data_path}")
    
    # Create dataset
    dataset = tf.data.Dataset.from_tensor_slices(image_files)
    dataset = dataset.map(load_fn, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(TrainingConfig.BUFFER_SIZE)
    dataset = dataset.batch(TrainingConfig.BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

def generator_loss(disc_generated_output, gen_output, target, LAMBDA=100):
    """Generator loss function"""
    gan_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(disc_generated_output), disc_generated_output)
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    total_gen_loss = gan_loss + (LAMBDA * l1_loss)
    return total_gen_loss, gan_loss, l1_loss

def discriminator_loss(disc_real_output, disc_generated_output):
    """Discriminator loss function"""
    real_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(disc_real_output), disc_real_output)
    generated_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.zeros_like(disc_generated_output), disc_generated_output)
    total_disc_loss = real_loss + generated_loss
    return total_disc_loss

def generate_images(generator, test_input, tar, epoch, output_dir):
    """Generate and save sample images"""
    prediction = generator(test_input, training=True)
    
    plt.figure(figsize=(15, 5))
    
    display_list = [test_input[0], tar[0], prediction[0]]
    title = ['Input Image', 'Ground Truth', 'Predicted Image']
    
    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.title(title[i])
        # Getting the pixel values in the [0, 1] range to plot it.
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    
    plt.savefig(os.path.join(output_dir, f'image_at_epoch_{epoch:04d}.png'))
    plt.close()

def train_step(generator, discriminator, generator_optimizer, discriminator_optimizer, 
               real_image, target, epoch):
    """Single training step"""
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(real_image, training=True)
        
        disc_real_output = discriminator([real_image, target], training=True)
        disc_generated_output = discriminator([real_image, gen_output], training=True)
        
        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)
    
    generator_gradients = gen_tape.gradient(gen_total_loss, generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    
    generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))
    
    return gen_total_loss, gen_gan_loss, gen_l1_loss, disc_loss

def fit(generator, discriminator, generator_optimizer, discriminator_optimizer, 
        train_dataset, test_dataset, epochs, output_dir):
    """Training loop"""
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    images_dir = os.path.join(output_dir, TrainingConfig.IMAGES_DIR)
    models_dir = os.path.join(output_dir, TrainingConfig.MODELS_DIR)
    logs_dir = os.path.join(output_dir, TrainingConfig.LOGS_DIR)
    
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    # Log file
    log_file = os.path.join(logs_dir, f"training_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    
    # Get a sample from test dataset for visualization
    for example_input, example_target in test_dataset.take(1):
        pass
    
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        
        # Training
        for n, (input_image, target) in train_dataset.enumerate():
            gen_total_loss, gen_gan_loss, gen_l1_loss, disc_loss = train_step(
                generator, discriminator, generator_optimizer, discriminator_optimizer,
                input_image, target, epoch
            )
            
            if n % TrainingConfig.LOG_EVERY_N_STEPS == 0:
                print(f"Step {n}: Gen Loss: {gen_total_loss:.4f}, Disc Loss: {disc_loss:.4f}")
        
        # Generate sample images
        if (epoch + 1) % TrainingConfig.SAVE_IMAGES_EVERY == 0:
            generate_images(generator, example_input, example_target, epoch + 1, images_dir)
        
        # Save model
        if (epoch + 1) % TrainingConfig.SAVE_CHECKPOINTS_EVERY == 0:
            generator.save_weights(os.path.join(models_dir, f'generator_epoch_{epoch + 1}.h5'))
            discriminator.save_weights(os.path.join(models_dir, f'discriminator_epoch_{epoch + 1}.h5'))
        
        # Log progress
        with open(log_file, 'a') as f:
            f.write(f"Epoch {epoch + 1} completed - Gen Loss: {gen_total_loss:.4f}, Disc Loss: {disc_loss:.4f}\n")
        
        print(f"Epoch {epoch + 1} completed - Gen Loss: {gen_total_loss:.4f}, Disc Loss: {disc_loss:.4f}")

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train pix2pix GAN')
    parser.add_argument('--data_path', type=str, required=True, help='Path to training data')
    parser.add_argument('--epochs', type=int, default=TrainingConfig.EPOCHS, help='Number of training epochs')
    parser.add_argument('--output_dir', type=str, default=TrainingConfig.OUTPUT_DIR, help='Output directory')
    parser.add_argument('--learning_rate', type=float, default=TrainingConfig.LEARNING_RATE, help='Learning rate')
    
    args = parser.parse_args()
    
    # Set up GPU memory growth
    if EnvironmentConfig.ALLOW_MEMORY_GROWTH:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(e)
    
    # Create models
    generator = create_generator()
    discriminator = create_discriminator()
    
    # Create optimizers
    generator_optimizer = tf.keras.optimizers.Adam(args.learning_rate, beta_1=TrainingConfig.BETA_1, beta_2=TrainingConfig.BETA_2)
    discriminator_optimizer = tf.keras.optimizers.Adam(args.learning_rate, beta_1=TrainingConfig.BETA_1, beta_2=TrainingConfig.BETA_2)
    
    # Create datasets
    train_dataset = create_dataset(args.data_path, is_training=True)
    test_dataset = create_dataset(args.data_path, is_training=False)
    
    # Train
    fit(generator, discriminator, generator_optimizer, discriminator_optimizer,
        train_dataset, test_dataset, args.epochs, args.output_dir)
    
    print("Training completed!")

if __name__ == "__main__":
    main() 