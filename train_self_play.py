"""
Self-play training where the model plays against itself.
14 games: 7 as white, 7 as black
Training stops when model reaches 100% win rate against itself.
"""

import torch
import numpy as np
from datetime import datetime
import os
from pathlib import Path
import sys

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import ChessNet
from chess_env import ChessEnvironment
from trainer import PPOTrainer
from config import (
    USE_CUDA, LEARNING_RATE, GAMMA, BATCH_SIZE, PPO_EPOCHS,
    CHECKPOINT_DIR, USE_CHESS_KNOWLEDGE
)


def run_self_play_training(max_epochs=100000, num_white_games=7, num_black_games=7):
    """
    Run self-play training where model plays against itself.
    
    Args:
        max_epochs: Maximum number of training epochs
        num_white_games: Number of games to play as white per epoch
        num_black_games: Number of games to play as black per epoch
    """
    
    print("=" * 80)
    print("CHESS ENGINE SELF-PLAY TRAINING")
    print("=" * 80)
    
    # Setup device
    device = torch.device('cuda' if USE_CUDA and torch.cuda.is_available() else 'cpu')
    if USE_CUDA and torch.cuda.is_available():
        print(f"\nUsing device: {device}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    else:
        print(f"\nUsing device: {device}")
    
    # Initialize model
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Initializing neural network...")
    model = ChessNet(num_residual_blocks=10, num_channels=512)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Model initialized with {total_params:,} trainable parameters")
    
    model = model.to(device)
    
    # Initialize trainer
    trainer = PPOTrainer(model, device=device, learning_rate=LEARNING_RATE)
    
    # Training configuration
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Training Configuration:")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]   - Self-play games per epoch: {num_white_games + num_black_games}")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]   - Games as white: {num_white_games}")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]   - Games as black: {num_black_games}")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]   - Max epochs: {max_epochs}")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]   - Learning rate: {LEARNING_RATE}")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]   - Chess knowledge enabled: {USE_CHESS_KNOWLEDGE}")
    if USE_CHESS_KNOWLEDGE:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]     • Opening book will be used for first moves")
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]     • Endgame knowledge will assist in endgame positions")
    
    # Check for checkpoint
    checkpoint_path = os.path.join(CHECKPOINT_DIR, 'self_play_latest_checkpoint.pt')
    if os.path.exists(checkpoint_path):
        response = input("\nFound existing self-play checkpoint. Load it? (y/n): ")
        if response.lower() == 'y':
            checkpoint = torch.load(checkpoint_path, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Checkpoint loaded")
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Resumed training from epoch {start_epoch}")
        else:
            start_epoch = 1
    else:
        start_epoch = 1
    
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting self-play training...")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] " + "=" * 80)
    
    # Training loop
    for epoch in range(start_epoch, max_epochs + 1):
        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] EPOCH {epoch}/{max_epochs}")
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] " + "-" * 80)
        
        # Phase 1: Play self-play games
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Phase 1: Playing self-play games...")
        
        from self_play_opponent import SelfPlayGameWorker
        import threading
        from queue import Queue
        
        results_queue = Queue()
        threads = []
        env = ChessEnvironment()
        
        # Play games as white
        for i in range(num_white_games):
            worker = SelfPlayGameWorker(i, model, device, use_knowledge=USE_CHESS_KNOWLEDGE)
            env_white = ChessEnvironment()
            thread = threading.Thread(
                target=lambda w=worker, e=env_white: results_queue.put(w.play_game(e, play_as_white=True))
            )
            threads.append(thread)
            thread.start()
        
        # Play games as black
        for i in range(num_black_games):
            worker = SelfPlayGameWorker(num_white_games + i, model, device, use_knowledge=USE_CHESS_KNOWLEDGE)
            env_black = ChessEnvironment()
            thread = threading.Thread(
                target=lambda w=worker, e=env_black: results_queue.put(w.play_game(e, play_as_white=False))
            )
            threads.append(thread)
            thread.start()
        
        # Wait for all games to complete
        for thread in threads:
            thread.join()
        
        # Collect results
        all_experiences = []
        wins = 0
        losses = 0
        draws = 0
        total_moves = 0
        
        while not results_queue.empty():
            game_result = results_queue.get()
            all_experiences.extend(game_result['experiences'])
            total_moves += game_result['moves']
            
            if game_result['result'] == 'Win':
                wins += 1
            elif game_result['result'] == 'Loss':
                losses += 1
            else:
                draws += 1
        
        games_played = wins + losses + draws
        win_rate = (wins / games_played * 100) if games_played > 0 else 0
        
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Games completed")
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]   Games played: {games_played}")
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]   Total moves: {total_moves}")
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]   Results - Wins: {wins}, Draws: {draws}, Losses: {losses}")
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]   Win Rate: {win_rate:.1f}%")
        
        # Check if model has achieved 100% win rate
        if win_rate == 100.0:
            print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] *** MODEL ACHIEVED 100% WIN RATE ***")
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Training complete!")
            
            # Save final checkpoint
            final_checkpoint_path = os.path.join(CHECKPOINT_DIR, 'self_play_final_model.pt')
            trainer.save_checkpoint(final_checkpoint_path, epoch, {
                'win_rate': win_rate,
                'final_epoch': epoch,
                'total_games': games_played
            })
            
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Final model saved to {final_checkpoint_path}")
            break
        
        # Phase 2: Train neural network
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Phase 2: Training neural network...")
        
        # Prepare training data
        training_data = {
            'states': [],
            'actions': [],
            'returns': [],
            'log_probs': [],
            'advantages': []
        }
        
        for exp in all_experiences:
            training_data['states'].append(exp['state'])
            training_data['actions'].append(exp['action'])
            training_data['returns'].append(exp['reward'])
            training_data['log_probs'].append(exp['log_prob'])
            training_data['advantages'].append(exp['reward'])
        
        # Train
        results = trainer.train_epoch(training_data, batch_size=BATCH_SIZE, ppo_epochs=PPO_EPOCHS)
        
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Training completed")
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]   Policy Loss: {results['policy_loss']:.6f}")
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]   Value Loss: {results['value_loss']:.6f}")
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]   Total Loss: {results['total_loss']:.6f}")
        
        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            trainer.save_checkpoint(checkpoint_path, epoch, {
                'win_rate': win_rate,
                'epoch': epoch
            })
        
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Epoch {epoch} completed")
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] " + "=" * 80)


if __name__ == "__main__":
    run_self_play_training(
        max_epochs=100000,
        num_white_games=7,
        num_black_games=7
    )
