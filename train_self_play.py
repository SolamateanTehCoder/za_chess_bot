"""
Self-play training where the model plays against itself.
28 games: 14 as white, 14 as black
Training stops when model reaches 100.0 accuracy (100% win rate against itself).

Each move is analyzed by Stockfish for accuracy-based rewards:
- Green timer flash = move received reward (good move)
- Red timer flash = move received pain penalty (bad move)
- Time penalty: 1 second baseline; each millisecond over incurs pain
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
from trainer import ChessTrainer
from stockfish_reward_analyzer import StockfishRewardAnalyzer
from game_visualizer import launch_visualizer
from config import (
    USE_CUDA, LEARNING_RATE, GAMMA, BATCH_SIZE, NUM_EPOCHS,
    CHECKPOINT_DIR, USE_CHESS_KNOWLEDGE
)


def run_self_play_training(max_epochs=100000, num_white_games=14, num_black_games=14):
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
    
    # Load Stockfish-trained model as starting point
    stockfish_checkpoint_path = os.path.join(CHECKPOINT_DIR, 'latest_checkpoint.pt')
    if os.path.exists(stockfish_checkpoint_path):
        try:
            checkpoint = torch.load(stockfish_checkpoint_path, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Loaded Stockfish-trained model as base for self-play")
        except Exception as e:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Warning: Could not load Stockfish model: {e}")
    else:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Warning: Stockfish checkpoint not found at {stockfish_checkpoint_path}")
    
    # Initialize trainer
    trainer = ChessTrainer(model, device=device, learning_rate=LEARNING_RATE)
    
    # Initialize Stockfish reward analyzer
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Initializing Stockfish reward analyzer...")
    reward_analyzer = StockfishRewardAnalyzer(depth=15, timeout_ms=300)
    if reward_analyzer.is_active:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ✓ Stockfish analyzer active - accuracy-based rewards enabled")
    else:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ⚠ Using fallback heuristic rewards (Stockfish not available)")
    
    # Launch game visualizer
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Launching real-time game visualizer...")
    try:
        visualizer = launch_visualizer(num_games=num_white_games + num_black_games)
        visualizer.set_status("Waiting for epoch to start...")
    except Exception as e:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Warning: Could not launch visualizer: {e}")
        visualizer = None
    
    # Training configuration
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Training Configuration:")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]   - Self-play games per epoch: {num_white_games + num_black_games} (Bullet: 60s per side)")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]   - Games as white: {num_white_games}")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]   - Games as black: {num_black_games}")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]   - Time control: 60 seconds per player (timeout = loss)")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]   - Move time baseline: 1 second (pain penalty per extra millisecond)")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]   - Reward system:")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]     • Green flash = move reward (good move by Stockfish analysis)")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]     • Red flash = move pain penalty (bad move, accuracy loss)")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]   - Max epochs: {max_epochs}")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]   - Learning rate: {LEARNING_RATE}")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]   - Chess knowledge enabled: {USE_CHESS_KNOWLEDGE}")
    if USE_CHESS_KNOWLEDGE:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]     • Opening book: 500+ openings (Sicilian, Ruy Lopez, Italian, French, Caro-Kann, etc.)")
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]     • Tactics: 19 patterns (pins, forks, skewers, discovered attacks, etc.)")
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]     • Strategy: 40+ concepts (control center, develop pieces, king safety, etc.)")
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]     • Endgame: 31 principles (opposition, zugzwang, king activity, etc.)")
    
    # Check for self-play checkpoint to resume
    self_play_checkpoint_path = os.path.join(CHECKPOINT_DIR, 'self_play_latest_checkpoint.pt')
    start_epoch = 1
    if os.path.exists(self_play_checkpoint_path):
        try:
            checkpoint = torch.load(self_play_checkpoint_path, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Loaded checkpoint from epoch {checkpoint['epoch']}")
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Resumed training from epoch {start_epoch}")
        except Exception as e:
            print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Warning: Could not load checkpoint: {e}")
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting fresh training")
    
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
        import time as time_module
        
        def play_game_worker(game_id, worker, env, play_as_white, queue):
            """Worker function for thread-safe game execution."""
            try:
                result = worker.play_game(env, play_as_white)
                queue.put(result)
            except Exception as e:
                print(f"Error in game {game_id}: {e}")
                queue.put(None)
        
        results_queue = Queue()
        threads = []
        game_start_time = time_module.time()
        
        # Play games as white
        for i in range(num_white_games):
            worker = SelfPlayGameWorker(i, model, device, use_knowledge=USE_CHESS_KNOWLEDGE, reward_analyzer=reward_analyzer)
            env_white = ChessEnvironment()
            thread = threading.Thread(
                target=play_game_worker,
                args=(i, worker, env_white, True, results_queue)
            )
            threads.append(thread)
            thread.start()
        
        # Play games as black
        for i in range(num_black_games):
            worker = SelfPlayGameWorker(num_white_games + i, model, device, use_knowledge=USE_CHESS_KNOWLEDGE, reward_analyzer=reward_analyzer)
            env_black = ChessEnvironment()
            thread = threading.Thread(
                target=play_game_worker,
                args=(num_white_games + i, worker, env_black, False, results_queue)
            )
            threads.append(thread)
            thread.start()
        
        # Wait for all games to complete, updating visualizer
        for thread in threads:
            thread.join()
        
        game_elapsed = time_module.time() - game_start_time
        
        # Collect results
        all_experiences = []
        wins = 0
        losses = 0
        draws = 0
        timeouts = 0
        total_moves = 0
        all_white_accuracies = []
        all_black_accuracies = []
        
        while not results_queue.empty():
            game_result = results_queue.get()
            if game_result is None:
                continue
            
            all_experiences.extend(game_result['experiences'])
            total_moves += game_result['moves']
            
            # Collect accuracies
            all_white_accuracies.extend(game_result.get('white_accuracies', []))
            all_black_accuracies.extend(game_result.get('black_accuracies', []))
            
            # Count result - timeout counts as a loss
            if game_result.get('timeout', False):
                timeouts += 1
                losses += 1
            elif game_result['result'] == 'Win':
                wins += 1
            elif game_result['result'] == 'Loss':
                losses += 1
            else:
                draws += 1
        
        games_played = wins + losses + draws
        win_rate = (wins / games_played * 100) if games_played > 0 else 0
        avg_white_accuracy = np.mean(all_white_accuracies) if all_white_accuracies else 0.0
        avg_black_accuracy = np.mean(all_black_accuracies) if all_black_accuracies else 0.0
        
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Games completed in {game_elapsed:.1f}s")
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]   Games played: {games_played}")
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]   Total moves: {total_moves}")
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]   Results - Wins: {wins}, Draws: {draws}, Losses: {losses} (Timeouts: {timeouts})")
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]   Win Rate: {win_rate:.1f}%")
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]   Move Accuracy - White: {avg_white_accuracy:.1f}%, Black: {avg_black_accuracy:.1f}%")
        
        # Update visualizer
        if visualizer:
            visualizer.set_epoch(epoch)
            visualizer.set_status(f"Epoch {epoch} | Win Rate: {win_rate:.1f}% | Accuracy: W:{avg_white_accuracy:.1f}% B:{avg_black_accuracy:.1f}% | Games: {games_played}")
        
        # Check if model has achieved 100.0 accuracy
        if win_rate == 100.0:
            print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] *** MODEL ACHIEVED 100.0 ACCURACY (100% WIN RATE) ***")
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
        results = trainer.train_epoch(training_data, batch_size=BATCH_SIZE, ppo_epochs=4)
        
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Training completed")
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]   Policy Loss: {results['policy_loss']:.6f}")
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]   Value Loss: {results['value_loss']:.6f}")
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]   Total Loss: {results['total_loss']:.6f}")
        
        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            trainer.save_checkpoint(self_play_checkpoint_path, epoch, {
                'win_rate': win_rate,
                'epoch': epoch
            })
        
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Epoch {epoch} completed")
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] " + "=" * 80)
    
    # Cleanup
    if reward_analyzer:
        reward_analyzer.close()
    if visualizer:
        visualizer.stop()


if __name__ == "__main__":
    run_self_play_training(
        max_epochs=100000,
        num_white_games=14,
        num_black_games=14
    )
