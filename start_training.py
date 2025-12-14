"""
Start Training Script for Za Chess Bot.

This script:
1. Verifies system requirements (Stockfish, directories).
2. Initializes the comprehensive training pipeline.
3. Starts the training loop.
"""

import os
import sys
import argparse
from pathlib import Path
import shutil

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from comprehensive_trainer import ComprehensiveStrategyTrainer

def check_stockfish():
    """Verify Stockfish is available."""
    print("[INIT] Verifying Stockfish...")
    
    # Check common paths
    paths = [
        r"C:\stockfish\stockfish-windows-x86-64-avx2.exe",
        "stockfish"
    ]
    
    found = False
    for path in paths:
        if path == "stockfish":
            if shutil.which("stockfish"):
                print(f"[OK] Stockfish found in PATH")
                found = True
                break
        elif os.path.exists(path):
            print(f"[OK] Stockfish found at {path}")
            found = True
            break
            
    if not found:
        print("[ERROR] Stockfish not found!")
        print("Please ensure Stockfish is installed at C:\\stockfish\\stockfish-windows-x86-64-avx2.exe")
        print("Or add 'stockfish' to your system PATH.")
        return False
        
    return True

def setup_directories():
    """Ensure necessary directories exist."""
    dirs = ["checkpoints", "checkpoints_strategy", "logs"]
    for d in dirs:
        Path(d).mkdir(exist_ok=True)
        print(f"[INIT] Ensured directory exists: {d}")

def main():
    parser = argparse.ArgumentParser(description="Start Za Chess Bot Training")
    parser.add_argument("--games", type=int, default=50, help="Games per cycle")
    parser.add_argument("--cycles", type=int, default=1000, help="Number of cycles")
    parser.add_argument("--epochs", type=int, default=10, help="Epochs per cycle")
    args = parser.parse_args()
    
    print("="*80)
    print("ZA CHESS BOT - TRAINING LAUNCHER")
    print("="*80)
    
    if not check_stockfish():
        sys.exit(1)
        
    setup_directories()
    
    print("\n[START] Starting Comprehensive Strategy Trainer...")
    try:
        trainer = ComprehensiveStrategyTrainer()
        
        # Run training cycles
        for cycle in range(args.cycles):
            print(f"\n[CYCLE {cycle+1}/{args.cycles}]")
            trainer.train_cycle(
                num_games=args.games,
                epochs=args.epochs,
                save_checkpoint=True
            )
            
            # Save games backup periodically
            if (cycle + 1) % 5 == 0:
                trainer.save_games_to_file()
                
    except KeyboardInterrupt:
        print("\n[STOP] Training interrupted by user.")
        trainer.save_games_to_file()
        print("[INFO] Saved game data.")
    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        raise

if __name__ == "__main__":
    main()
