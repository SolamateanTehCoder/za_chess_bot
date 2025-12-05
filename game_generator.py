"""
Game Generator for GitHub Actions
Plays bullet chess games continuously and saves game data
"""
import json
import sys
import time
import torch
from datetime import datetime
from pathlib import Path

# Import from project
from game_player import BulletGamePlayer
from game_player import StockfishAnalyzer
from trainer import SimpleChessNet


class GameBatchLogger:
    """Log games to batch files for training"""
    
    def __init__(self, batch_dir="game_batches"):
        self.batch_dir = Path(batch_dir)
        self.batch_dir.mkdir(exist_ok=True)
        self.batch_file = None
        self.game_count = 0
        self.start_new_batch()
    
    def start_new_batch(self):
        """Start a new batch file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.batch_file = self.batch_dir / f"games_batch_{timestamp}.jsonl"
        self.game_count = 0
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Started new batch: {self.batch_file.name}")
    
    def log_game(self, game_data):
        """Append game to batch file"""
        with open(self.batch_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(game_data) + '\n')
        self.game_count += 1
    
    def get_batch_stats(self):
        """Get current batch statistics"""
        return {
            'file': self.batch_file.name,
            'games': self.game_count
        }


def generate_games(duration_hours=5, max_games=None):
    """
    Generate games for specified duration
    
    Args:
        duration_hours: How long to run (in hours)
        max_games: Maximum games to generate (optional)
    """
    print("=" * 80)
    print("BULLET CHESS GAME GENERATOR (GitHub Actions)")
    print("=" * 80)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Device: {device}")
    if device.type == 'cuda':
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] GPU: {torch.cuda.get_device_name(0)}")
    
    # Initialize model (load latest or create new)
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Initializing model...")
    model = SimpleChessNet(device=device)
    
    # Try to load latest checkpoint
    checkpoint_dir = Path("checkpoints")
    if checkpoint_dir.exists():
        game_checkpoints = list(checkpoint_dir.glob("model_checkpoint_game_*.pt"))
        if game_checkpoints:
            # Sort by game number numerically
            game_checkpoints.sort(key=lambda p: int(p.stem.split('_')[-1]))
            latest = game_checkpoints[-1]
            try:
                checkpoint = torch.load(latest, map_location='cpu')
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Loaded model from {latest.name}")
            except Exception as e:
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Could not load checkpoint: {e}")
    
    # Initialize game player
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Initializing Stockfish analyzer...")
    analyzer = StockfishAnalyzer()
    game_player = BulletGamePlayer(model, device=device, stockfish_analyzer=analyzer)
    
    # Initialize batch logger
    logger = GameBatchLogger()
    
    # Game generation loop
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting game generation...")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Duration: {duration_hours} hours")
    if max_games:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Max games: {max_games}")
    
    start_time = time.time()
    game_num = 0
    total_games = 0
    results = {'Win': 0, 'Loss': 0, 'Draw': 0}
    
    try:
        while True:
            # Check time limit
            elapsed_hours = (time.time() - start_time) / 3600
            if elapsed_hours >= duration_hours:
                print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Duration reached ({duration_hours}h)")
                break
            
            # Check game limit
            if max_games and game_num >= max_games:
                print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Game limit reached ({max_games})")
                break
            
            # Play game
            game_num += 1
            total_games += 1
            play_as_white = (game_num % 2 == 1)  # Alternate colors
            
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Game {game_num} ({elapsed_hours:.1f}h/{duration_hours}h)")
            
            game_start = time.time()
            result_data = game_player.play_game(play_as_white=play_as_white)
            game_duration = time.time() - game_start
            
            # Log game
            game_data = {
                'game_number': total_games,
                'timestamp': datetime.now().isoformat(),
                'ai_color': 'white' if play_as_white else 'black',
                'result': result_data['result'],
                'duration': game_duration,
                'total_moves': result_data['moves'],
                'ai_moves_count': result_data['num_ai_moves'],
                'moves': []
            }
            
            # Add move details
            for i, exp in enumerate(result_data['experiences']):
                game_data['moves'].append({
                    'move_number': i + 1,
                    'uci': exp['move'],
                    'reward': exp['reward'],
                    'stockfish_reward': exp['stockfish_reward'],
                    'time_penalty': exp['time_penalty'],
                    'move_time_ms': exp['move_time_ms']
                })
            
            logger.log_game(game_data)
            results[result_data['result']] += 1
            
            # Print stats every 10 games
            if game_num % 10 == 0:
                batch_stats = logger.get_batch_stats()
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Batch stats: {batch_stats['games']} games, Results: {results['Win']}W {results['Loss']}L {results['Draw']}D")
    
    except KeyboardInterrupt:
        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Interrupted by user")
    
    finally:
        if analyzer:
            analyzer.close()
        
        # Final stats
        elapsed_hours = (time.time() - start_time) / 3600
        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Game generation complete")
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Total games: {total_games}")
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Results: {results['Win']}W {results['Loss']}L {results['Draw']}D")
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Elapsed time: {elapsed_hours:.1f} hours")
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Games per hour: {total_games / elapsed_hours:.1f}")
        
        batch_stats = logger.get_batch_stats()
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Batch file: {batch_stats['file']}")


if __name__ == "__main__":
    # For GitHub Actions, run for 5 hours (or max 1000 games as safety limit)
    duration = 5  # hours
    max_games = 1000  # safety limit
    
    if len(sys.argv) > 1:
        duration = int(sys.argv[1])
    if len(sys.argv) > 2:
        max_games = int(sys.argv[2])
    
    generate_games(duration_hours=duration, max_games=max_games)
