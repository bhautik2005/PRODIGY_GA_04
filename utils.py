import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import json
import datetime
from pathlib import Path

def visualize_training_progress(logs_dir, output_path=None):
    """Visualize training progress from logs"""
    log_files = []
    for file in os.listdir(logs_dir):
        if file.endswith('.log'):
            log_files.append(os.path.join(logs_dir, file))
    
    if not log_files:
        print("No log files found")
        return
    
    # Read the latest log file
    latest_log = max(log_files, key=os.path.getctime)
    
    epochs = []
    gen_losses = []
    disc_losses = []
    
    with open(latest_log, 'r') as f:
        for line in f:
            if 'Epoch' in line and 'completed' in line:
                try:
                    # Parse epoch number and losses
                    parts = line.split(' - ')
                    epoch_part = parts[0].split('Epoch ')[-1].split(' completed')[0]
                    epoch = int(epoch_part)
                    
                    # Extract losses
                    gen_loss = float(line.split('Gen Loss: ')[1].split(',')[0])
                    disc_loss = float(line.split('Disc Loss: ')[1])
                    
                    epochs.append(epoch)
                    gen_losses.append(gen_loss)
                    disc_losses.append(disc_loss)
                except:
                    continue
    
    if not epochs:
        print("No training data found in logs")
        return
    
    # Create plot
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, gen_losses, label='Generator Loss', color='blue')
    plt.plot(epochs, disc_losses, label='Discriminator Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Generator vs Discriminator Loss')
    plt.legend()