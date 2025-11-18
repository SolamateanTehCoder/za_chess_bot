"""Quick test - play one game only."""
import sys
import torch
import os
from datetime import datetime

# Setup paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import USE_CUDA, USE_CHESS_KNOWLEDGE
from model import ChessNet
from chess_env import ChessEnvironment
from self_play_opponent import SelfPlayGameWorker

print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Loading model...")
device = torch.device('cuda' if USE_CUDA and torch.cuda.is_available() else 'cpu')

# Load model
model = ChessNet().to(device)
checkpoint_path = os.path.join('checkpoints', 'self_play_latest_checkpoint.pt')
checkpoint = torch.load(checkpoint_path, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Playing test game...")

# Create worker
worker = SelfPlayGameWorker(0, model, device, use_knowledge=USE_CHESS_KNOWLEDGE)

# Create environment
env = ChessEnvironment()

# Play one game
print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Game starting as white...")
result = worker.play_game(env, play_as_white=True)

print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Game result: {result['result']}")
print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Moves played: {result['moves']}")
print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Experiences collected: {len(result['experiences'])}")
print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Test complete!")
