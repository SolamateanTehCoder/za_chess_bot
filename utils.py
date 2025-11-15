"""Utility functions for the chess engine."""

import os
import json
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime


def plot_training_progress(training_stats, output_dir="plots"):
    """
    Plot comprehensive training progress including losses and accuracy.
    
    Args:
        training_stats: Dictionary with training statistics
        output_dir: Directory to save plots
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    policy_losses = training_stats.get('policy_losses', [])
    value_losses = training_stats.get('value_losses', [])
    total_losses = training_stats.get('total_losses', [])
    win_rates = training_stats.get('win_rates', [])
    draw_rates = training_stats.get('draw_rates', [])
    loss_rates = training_stats.get('loss_rates', [])
    
    if not policy_losses:
        print("No training data to plot")
        return
    
    epochs = range(1, len(policy_losses) + 1)
    
    # Create comprehensive figure with subplots
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Win/Loss/Draw Rates
    ax1 = plt.subplot(3, 2, 1)
    if win_rates:
        ax1.plot(epochs[:len(win_rates)], win_rates, 'g-', linewidth=2, label='Win Rate')
        ax1.plot(epochs[:len(draw_rates)], draw_rates, 'y-', linewidth=2, label='Draw Rate')
        ax1.plot(epochs[:len(loss_rates)], loss_rates, 'r-', linewidth=2, label='Loss Rate')
        ax1.axhline(y=100, color='k', linestyle='--', alpha=0.3, label='Target (100%)')
        ax1.set_title('Game Results Over Time', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Percentage (%)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 105)
    
    # 2. Win Rate Only (larger view)
    ax2 = plt.subplot(3, 2, 2)
    if win_rates:
        ax2.plot(epochs[:len(win_rates)], win_rates, 'g-', linewidth=3)
        ax2.axhline(y=100, color='k', linestyle='--', alpha=0.3, label='Target')
        ax2.fill_between(epochs[:len(win_rates)], 0, win_rates, alpha=0.3, color='green')
        ax2.set_title('Win Rate Progress', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Win Rate (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 105)
    
    # 3. Policy Loss
    ax3 = plt.subplot(3, 2, 3)
    ax3.plot(epochs, policy_losses, 'b-', linewidth=2)
    ax3.set_title('Policy Loss', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss')
    ax3.grid(True, alpha=0.3)
    
    # 4. Value Loss
    ax4 = plt.subplot(3, 2, 4)
    ax4.plot(epochs, value_losses, 'r-', linewidth=2)
    ax4.set_title('Value Loss', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Loss')
    ax4.grid(True, alpha=0.3)
    
    # 5. Total Loss
    ax5 = plt.subplot(3, 2, 5)
    ax5.plot(epochs, total_losses, 'purple', linewidth=2)
    ax5.set_title('Total Loss', fontsize=14, fontweight='bold')
    ax5.set_xlabel('Epoch')
    ax5.set_ylabel('Loss')
    ax5.grid(True, alpha=0.3)
    
    # 6. Moving Average Win Rate (smoothed)
    ax6 = plt.subplot(3, 2, 6)
    if win_rates and len(win_rates) > 5:
        window_size = min(10, len(win_rates) // 2)
        if window_size > 1:
            moving_avg = np.convolve(win_rates, np.ones(window_size)/window_size, mode='valid')
            ax6.plot(range(window_size, len(win_rates) + 1), moving_avg, 'g-', linewidth=3, label=f'{window_size}-epoch MA')
            ax6.plot(epochs[:len(win_rates)], win_rates, 'g-', alpha=0.3, linewidth=1, label='Actual')
            ax6.axhline(y=100, color='k', linestyle='--', alpha=0.3)
            ax6.set_title('Smoothed Win Rate', fontsize=14, fontweight='bold')
            ax6.set_xlabel('Epoch')
            ax6.set_ylabel('Win Rate (%)')
            ax6.legend()
            ax6.grid(True, alpha=0.3)
            ax6.set_ylim(0, 105)
    
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = os.path.join(output_dir, f"training_progress_{timestamp}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Training charts saved to {plot_path}")
    
    plt.close()


def save_training_stats(stats, filepath="training_stats.json"):
    """
    Save training statistics to a JSON file.
    
    Args:
        stats: Dictionary with training statistics
        filepath: Path to save the JSON file
    """
    with open(filepath, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Training statistics saved to {filepath}")


def load_training_stats(filepath="training_stats.json"):
    """
    Load training statistics from a JSON file.
    
    Args:
        filepath: Path to the JSON file
        
    Returns:
        Dictionary with training statistics
    """
    if not os.path.exists(filepath):
        return {}
    
    with open(filepath, 'r') as f:
        stats = json.load(f)
    
    return stats


def print_model_summary(model):
    """
    Print a summary of the model architecture.
    
    Args:
        model: PyTorch model
    """
    print("\nModel Summary:")
    print("=" * 80)
    
    total_params = 0
    trainable_params = 0
    
    for name, param in model.named_parameters():
        num_params = param.numel()
        total_params += num_params
        
        if param.requires_grad:
            trainable_params += num_params
        
        print(f"{name:50s} {str(param.shape):30s} {num_params:15,d}")
    
    print("=" * 80)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    print("=" * 80)
    print()


def format_time(seconds):
    """
    Format seconds into a human-readable string.
    
    Args:
        seconds: Number of seconds
        
    Returns:
        Formatted time string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"
