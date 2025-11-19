#!/usr/bin/env python3
"""
Quick Start: Run the chess engine with accuracy-based rewards and visualization.

This script demonstrates how to use the new reward system.
"""

import sys
import os

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    """Run the chess engine training with rewards and visualization."""
    
    print("=" * 80)
    print("CHESS BOT SELF-PLAY TRAINING WITH ACCURACY REWARDS")
    print("=" * 80)
    print()
    print("Features:")
    print("  ✓ Stockfish-based accuracy rewards for every move")
    print("  ✓ Real-time visualization of all 28 games")
    print("  ✓ Green timer flashes = move reward (good move)")
    print("  ✓ Red timer flashes = move pain (bad move)")
    print("  ✓ Time penalty: 1 second baseline, pain for extra milliseconds")
    print("  ✓ Comprehensive chess knowledge for both players")
    print("  ✓ Bullet time control (60 seconds per side)")
    print()
    print("=" * 80)
    print()
    
    # Import and run training
    from train_self_play import run_self_play_training
    
    print("Starting training...\n")
    
    try:
        run_self_play_training(
            max_epochs=100000,
            num_white_games=14,
            num_black_games=14
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
    except Exception as e:
        print(f"\n\nError during training: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
