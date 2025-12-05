"""
Bullet Chess Model Training System
Plays games until 100% accuracy, then trains the model.
One game at a time, 60 second time control per side.
Stockfish-based reward system with millisecond-based pain penalty.
Logs all games move-by-move and supports checkpoint recovery.
"""

import os
import sys
import torch
import numpy as np
from datetime import datetime
from pathlib import Path
import json

# PyTorch setup
os.environ['TORCH_COMPILE_DEBUG'] = '0'
torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)

try:
    torch.backends.cudnn.conv.fp32_precision = 'tf32'
    torch.backends.cuda.matmul.fp32_precision = 'tf32'
except:
    try:
        torch.set_float32_matmul_precision('high')
    except:
        pass

from game_player import BulletGamePlayer, StockfishAnalyzer
from trainer import SimpleChessNet, ChessTrainer


class GameLogger:
    """Logs games move-by-move to file."""
    
    def __init__(self, log_file: str = "games.jsonl"):
        """
        Initialize game logger.
        
        Args:
            log_file: Path to append-only log file (JSONL format)
        """
        self.log_file = Path(log_file)
        self.game_count = 0
    
    def log_game(self, game_num: int, play_as_white: bool, result: dict, duration: float):
        """
        Log a complete game move-by-move.
        
        Args:
            game_num: Game number
            play_as_white: Whether AI played as white
            result: Game result dict with moves and experiences
            duration: Game duration in seconds
        """
        game_data = {
            'game_number': game_num,
            'timestamp': datetime.now().isoformat(),
            'ai_color': 'white' if play_as_white else 'black',
            'result': result['result'],
            'duration': duration,
            'total_moves': result['moves'],
            'ai_moves_count': result['num_ai_moves'],
            'moves': []
        }
        
        # Add move details
        for i, exp in enumerate(result['experiences']):
            game_data['moves'].append({
                'move_number': i + 1,
                'uci': exp['move'],
                'reward': exp['reward'],
                'stockfish_reward': exp['stockfish_reward'],
                'time_penalty': exp['time_penalty'],
                'move_time_ms': exp['move_time_ms']
            })
        
        # Append to JSONL file
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(game_data) + '\n')
        
        self.game_count += 1


def load_training_state(checkpoint_dir: Path):
    """
    Load training state from latest checkpoint if available.
    
    Returns:
        Tuple of (total_games_so_far, checkpoint_dict) or (0, None)
    """
    if not checkpoint_dir.exists():
        return 0, None
    
    # Find latest checkpoint - prefer game checkpoints (self-play), fallback to epoch checkpoints
    game_checkpoints = list(checkpoint_dir.glob("model_checkpoint_game_*.pt"))
    epoch_checkpoints = sorted(checkpoint_dir.glob("model_after_100pct_epoch_*.pt"))
    
    if game_checkpoints:
        # Sort by game number numerically, not lexicographically
        game_checkpoints.sort(key=lambda p: int(p.stem.split('_')[-1]))
        latest = game_checkpoints[-1]
    elif epoch_checkpoints:
        latest = epoch_checkpoints[-1]
    else:
        return 0, None
    
    try:
        checkpoint = torch.load(latest, map_location='cpu')
        # Extract metadata from nested structure
        metadata = checkpoint.get('metadata', checkpoint)  # Fallback to top-level if no metadata key
        total_games = metadata.get('total_games_played', 0)
        epoch = metadata.get('epoch', 0)
        
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Found checkpoint: {latest.name}")
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]   Training epoch: {epoch}, Total games: {total_games}")
        
        return total_games, checkpoint
    except Exception as e:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Error loading checkpoint: {e}")
        return 0, None


def main():
    """Main training loop."""
    print("=" * 80)
    print("BULLET CHESS TRAINING SYSTEM")
    print("=" * 80)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Device: {device}")
    if device.type == 'cuda':
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] CUDA Version: {torch.version.cuda}")
    
    # Create checkpoints directory
    checkpoint_dir = Path('checkpoints')
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Initialize game logger
    logger = GameLogger("games.jsonl")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Game log: games.jsonl")
    
    # Try to load from checkpoint
    total_games_completed, checkpoint = load_training_state(checkpoint_dir)
    training_epoch = 0
    if checkpoint:
        training_epoch = checkpoint.get('epoch', 0) + 1
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Resuming from epoch {training_epoch}")
    
    # Initialize model
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Initializing neural network...")
    model = SimpleChessNet()
    
    # Load checkpoint if available
    if checkpoint:
        try:
            # Try both naming conventions
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif 'model_state' in checkpoint:
                model.load_state_dict(checkpoint['model_state'])
            else:
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Warning: No model state found in checkpoint")
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Model loaded from checkpoint")
        except Exception as e:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Could not load model state from checkpoint: {e}")
    
    trainer = ChessTrainer(model, device=device, learning_rate=0.001)
    game_player = BulletGamePlayer(model, device=device, stockfish_analyzer=StockfishAnalyzer())
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Model parameters: {total_params:,}")
    
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Game Configuration:")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]   - Time control: 60 seconds per side (Bullet)")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]   - Continuous thinking: Yes (thinks during opponent's time)")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]   - Time penalty: -0.001 per millisecond")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]   - Reward source: Stockfish analysis")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]   - Game format: Single game at a time")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]   - Training starts: After 100% accuracy achieved")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]   - Logging: games.jsonl (move-by-move)")
    
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] " + "=" * 80)
    
    # Game playing phase
    # Initialize from checkpoint if available
    if checkpoint:
        metadata = checkpoint.get('metadata', checkpoint)  # Fallback to top-level
        total_games_completed = metadata.get('total_games_played', 0)
        white_wins = metadata.get('white_wins', 0)
        black_wins = metadata.get('black_wins', 0)
        draws = metadata.get('draws', 0)
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Resuming from game {total_games_completed + 1}")
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]   Previous stats: {white_wins}W {black_wins}L {draws}D")
    else:
        total_games_completed = 0
        white_wins = 0
        black_wins = 0
        draws = 0
    
    game_count = 0
    self_play_count = 0  # Counts self-play games (model vs model)
    game_experiences = []
    
    try:
        while True:
            self_play_count += 1
            overall_game_num = total_games_completed + self_play_count
            
            print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] SELF-PLAY GAME {overall_game_num}")
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Model (White) vs Model (Black)")
            
            # Play self-play game (model vs itself)
            game_start = datetime.now()
            result_white = game_player.play_game(play_as_white=True)
            game_duration = (datetime.now() - game_start).total_seconds()
            
            # Log the game from white's perspective
            logger.log_game(overall_game_num, True, result_white, game_duration)
            
            # Track result
            if result_white['result'] == 'Win':
                white_wins += 1
            elif result_white['result'] == 'Draw':
                draws += 1
            else:
                black_wins += 1
            
            game_experiences.append(result_white)
            game_count += 1
            
            # Print game summary
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Result: {result_white['result']}")
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Duration: {game_duration:.1f}s")
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] AI moves: {result_white['num_ai_moves']}/{result_white['moves']}")
            
            if result_white['experiences']:
                rewards = [exp['reward'] for exp in result_white['experiences']]
                avg_reward = np.mean(rewards)
                times = [exp['move_time_ms'] for exp in result_white['experiences']]
                avg_time = np.mean(times)
                
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Avg reward (White): {avg_reward:.4f}")
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Avg move time: {avg_time:.1f}ms")
            
            # Calculate statistics
            total_games_played = white_wins + black_wins + draws
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Self-Play Stats: {white_wins}W {black_wins}L {draws}D ({total_games_played} games)")
            
            sys.stdout.flush()
            
            # Save checkpoint every 500 games
            if overall_game_num % 500 == 0:
                checkpoint_path = checkpoint_dir / f"model_checkpoint_game_{overall_game_num}.pt"
                trainer.save_checkpoint(str(checkpoint_path), {
                    'game_count': game_count,
                    'total_games_played': overall_game_num,
                    'white_wins': white_wins,
                    'black_wins': black_wins,
                    'draws': draws,
                    'epoch': training_epoch
                })
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] *** CHECKPOINT SAVED: {checkpoint_path.name} ***")
    
    except KeyboardInterrupt:
        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Training interrupted by user")
    
    finally:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Closing...")
        if hasattr(game_player, 'analyzer') and game_player.analyzer:
            game_player.analyzer.close()



if __name__ == "__main__":
    main()
