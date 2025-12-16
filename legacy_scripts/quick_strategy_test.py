"""
Quick Strategy Training Test - See all strategies in action.
Generates a few games with each strategy pairing and trains from both sides.
"""

import torch
from hybrid_player import HybridChessPlayer
from comprehensive_trainer import ComprehensiveStrategyTrainer
from chess_strategies import ChessStrategy


def main():
    print("\n" + "="*80)
    print("QUICK STRATEGY TRAINING TEST")
    print("="*80)
    
    # Check CUDA
    print(f"\n[GPU] CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[GPU] Device: {torch.cuda.get_device_name(0)}")
    
    # Initialize trainer
    print("\n[INIT] Loading player and initializing trainer...")
    trainer = ComprehensiveStrategyTrainer()
    
    print(f"[STRATEGIES] Available strategies: {len(list(ChessStrategy))}")
    for strat in ChessStrategy:
        print(f"  - {strat.value}")
    
    # Run quick training cycle
    print("\n[START] Running quick training cycle with 10 games...\n")
    
    stats = trainer.train_cycle(
        num_games=10,
        epochs=3,
        save_checkpoint=True
    )
    
    # Save games
    trainer.save_games_to_file()
    
    # Print strategy statistics
    print("\n" + "="*80)
    print("STRATEGY PERFORMANCE SUMMARY")
    print("="*80)
    
    if trainer.strategy_stats:
        print("\nStrategy Win Rates:")
        for strat in sorted(trainer.strategy_stats.keys()):
            s = trainer.strategy_stats[strat]
            if s["games_played"] > 0:
                win_rate = s["wins"] / s["games_played"]
                print(f"  {strat:15s}: {win_rate:.1%} ({s['wins']}/{s['games_played']})")
    
    print("\n" + "="*80)
    print("[DONE] Quick test complete! Check comprehensive_strategy_games.jsonl\n")


if __name__ == "__main__":
    main()
