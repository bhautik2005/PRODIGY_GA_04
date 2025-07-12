#!/usr/bin/env python3
"""
Configuration file for pix2pix GAN training
Modify these settings to customize training parameters
"""

import os

class TrainingConfig:
    """Training configuration parameters"""
    
    # Dataset Configuration
    DATASET_NAME = 'facades'  # Options: 'facades', 'cityscapes', 'maps', 'edges2shoes'
    DATASET_URLS = {
        'facades': 'https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/facades.tar.gz',
        'cityscapes': 'https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/cityscapes.tar.gz',
        'maps': 'https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/maps.tar.gz',
        'edges2shoes': 'https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/edges2shoes.tar.gz'
    }
    
    # Image Configuration
    IMG_HEIGHT = 256
    IMG_WIDTH = 256
    IMG_CHANNELS = 3
    
    # Data Loading Configuration
    BUFFER_SIZE = 400
    BATCH_SIZE = 1  # pix2pix typically uses batch size 1
    PREFETCH_BUFFER = 1
    
    # Training Configuration
    EPOCHS = 200
    LEARNING_RATE = 2e-4
    BETA_1 = 0.5  # Adam optimizer beta1 parameter
    BETA_2 = 0.999  # Adam optimizer beta2 parameter
    
    # Loss Configuration
    LAMBDA_L1 = 100  # Weight for L1 loss in generator
    LAMBDA_GAN = 1   # Weight for adversarial loss
    
    # Model Configuration
    GENERATOR_FILTERS = [64, 128, 256, 512, 512, 512, 512, 512]
    DISCRIMINATOR_FILTERS = [64, 128, 256, 512]
    
    # Regularization
    DROPOUT_RATE = 0.5
    APPLY_DROPOUT_LAYERS = [5, 6, 7]  # Which decoder layers to apply dropout
    
    # Checkpoint Configuration
    SAVE_CHECKPOINTS_EVERY = 20  # Save model every N epochs
    MAX_CHECKPOINTS_TO_KEEP = 5
    
    # Logging Configuration
    LOG_EVERY_N_STEPS = 50
    SAVE_IMAGES_EVERY = 1  # Save sample images every N epochs
    
    # Output Directories
    OUTPUT_DIR = 'pix2pix_output'
    IMAGES_DIR = 'generated_images'
    MODELS_DIR = 'saved_models'
    LOGS_DIR = 'logs'
    TENSORBOARD_DIR = 'tensorboard_logs'

class ModelConfig:
    """Model architecture configuration"""
    
    # Generator Configuration (U-Net)
    GENERATOR_NAME = 'UNetGenerator'
    ENCODER_LAYERS = 8
    DECODER_LAYERS = 7
    SKIP_CONNECTIONS = True
    
    # Discriminator Configuration (PatchGAN)
    DISCRIMINATOR_NAME = 'PatchGANDiscriminator'
    PATCH_SIZE = 70  # Size of patches the discriminator classifies
    N_LAYERS = 3     # Number of layers in discriminator
    
    # Weight Initialization
    WEIGHT_INIT_MEAN = 0.0
    WEIGHT_INIT_STD = 0.02

class DataAugmentationConfig:
    """Data augmentation configuration"""
    
    # Augmentation settings
    APPLY_RANDOM_JITTER = True
    JITTER_RESIZE_TO = 286  # Resize to this size before cropping
    RANDOM_CROP_SIZE = 256  # Final crop size
    
    APPLY_RANDOM_FLIP = True
    FLIP_PROBABILITY = 0.5
    
    # Color augmentation (optional)
    APPLY_COLOR_JITTER = False
    BRIGHTNESS_DELTA = 0.1
    CONTRAST_RANGE = (0.8, 1.2)
    HUE_DELTA = 0.1
    SATURATION_RANGE = (0.8, 1.2)

class InferenceConfig:
    """Configuration for inference/testing"""
    
    # Input/Output settings
    INPUT_SIZE = (256, 256)
    OUTPUT_FORMAT = 'PNG'  # Options: 'PNG', 'JPEG'
    SAVE_COMPARISON = True  # Save input-output comparison
    
    # Batch inference settings
    BATCH_SIZE = 4
    USE_GPU = True

# Environment Configuration
class EnvironmentConfig:
    """Environment and system configuration"""
    
    # GPU Configuration
    ENABLE_GPU = True
    GPU_MEMORY_LIMIT = None  # Set to specific value (MB) to limit GPU memory
    ALLOW_MEMORY_GROWTH = True
    
    # Multi-GPU settings
    USE_MIXED_PRECISION = False  # Enable for faster training on modern GPUs
    DISTRIBUTION_STRATEGY = None  # Options: None, 'mirrored', 'multi_worker'
    
    # Reproducibility
    RANDOM_SEED = 42
    SET_DETERMINISTIC = False  # Set to True for fully deterministic training
    
    # Performance
    PARALLEL_CALLS = -1  # Use all available cores for data processing
    PREFETCH_BUFFER_SIZE = 2

# Validation Configuration
class ValidationConfig:
    """Validation and monitoring configuration"""
    
    # Validation settings
    VALIDATION_SPLIT = 0.1  # Fraction of training data to use for validation
    VALIDATE_EVERY_N_EPOCHS = 5
    
    # Metrics to track
    TRACK_METRICS = ['generator_loss', 'discriminator_loss', 'l1_loss', 'gan_loss']
    
    # Early stopping
    ENABLE_EARLY_STOPPING = False
    EARLY_STOPPING_PATIENCE = 10
    EARLY_STOPPING_MONITOR = 'generator_loss'
    
    # Learning rate scheduling
    ENABLE_LR_SCHEDULING = False
    LR_SCHEDULE_TYPE = 'exponential'  # Options: 'exponential', 'cosine', 'polynomial'
    LR_DECAY_RATE = 0.95
    LR_DECAY_STEPS = 1000

# Export configurations
def get_config(config_type='training'):
    """Get configuration based on type"""
    configs = {
        'training': TrainingConfig(),
        'model': ModelConfig(),
        'augmentation': DataAugmentationConfig(),
        'inference': InferenceConfig(),
        'environment': EnvironmentConfig(),
        'validation': ValidationConfig()
    }
    
    if config_type == 'all':
        return configs
    
    return configs.get(config_type, TrainingConfig())

# Default configuration
DEFAULT_CONFIG = {
    'training': TrainingConfig(),
    'model': ModelConfig(),
    'augmentation': DataAugmentationConfig(),
    'inference': InferenceConfig(),
    'environment': EnvironmentConfig(),
    'validation': ValidationConfig()
}