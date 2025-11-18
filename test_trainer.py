"""Test trainer with sample data."""
import torch
import sys
import os

os.chdir('c:\\Arjun\\Za Chess Bot')
sys.path.insert(0, os.getcwd())

from model import ChessNet
from trainer import ChessTrainer
from config import USE_CUDA, BATCH_SIZE

print("Creating model and trainer...")
device = torch.device('cuda' if USE_CUDA and torch.cuda.is_available() else 'cpu')
model = ChessNet().to(device)
trainer = ChessTrainer(model, device=device)

print("Creating sample training data...")
# Create 10 sample experiences
training_data = {
    'states': [torch.randn(119, 8, 8) for _ in range(10)],
    'actions': [i % 4672 for i in range(10)],
    'returns': [0.5] * 10,
    'log_probs': [0.1] * 10,
    'advantages': [0.2] * 10
}

print("Starting training...")
try:
    results = trainer.train_epoch(training_data, batch_size=4, ppo_epochs=1)
    print(f"Training completed!")
    print(f"Results: {results}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
