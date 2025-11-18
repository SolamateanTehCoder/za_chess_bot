"""Minimal test - just initialize model."""
import sys
import torch
import os
from datetime import datetime

os.chdir('c:\\Arjun\\Za Chess Bot')
sys.path.insert(0, os.getcwd())

print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Initializing model...")
from model import ChessNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Device: {device}")

print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Creating model...")
model = ChessNet().to(device)
print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Model created")

print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Loading checkpoint...")
checkpoint_path = os.path.join('checkpoints', 'self_play_latest_checkpoint.pt')
print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Checkpoint path: {checkpoint_path}")
print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] File exists: {os.path.exists(checkpoint_path)}")

checkpoint = torch.load(checkpoint_path, weights_only=False)
print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Checkpoint loaded")

print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Loading state dict...")
model.load_state_dict(checkpoint['model_state_dict'])
print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] State dict loaded")

model.eval()
print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Model set to eval mode")

print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Done!")
