"""Main training script for the chess engine."""

import os
import time
import torch
from datetime import datetime
from model import ChessNet
from trainer import ChessTrainer
from parallel_player import ParallelGamePlayer
from utils import plot_training_progress
from config import (
    NUM_PARALLEL_GAMES, NUM_EPOCHS, LOOKAHEAD_MOVES,
    HIDDEN_SIZE, NUM_RESIDUAL_BLOCKS, LEARNING_RATE,
    USE_CUDA, CHECKPOINT_DIR, SAVE_FREQUENCY, LOG_FILE, TARGET_WIN_RATE, USE_CHESS_KNOWLEDGE
)


def setup_directories():
    """Create necessary directories."""
    dirs = [CHECKPOINT_DIR, "plots"]
    for directory in dirs:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")


def log_message(message, log_file=LOG_FILE):
    """Log a message to console and file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    print(log_entry)
    
    with open(log_file, 'a') as f:
        f.write(log_entry + '\n')


def main():
    """Main training loop."""
    print("=" * 80)
    print("CHESS ENGINE REINFORCEMENT LEARNING TRAINING")
    print("=" * 80)
    print()
    
    # Setup
    setup_directories()
    
    # Device configuration
    device = torch.device("cuda" if USE_CUDA and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    print()
    
    # Initialize model
    log_message("Initializing neural network...")
    model = ChessNet(
        num_residual_blocks=NUM_RESIDUAL_BLOCKS,
        num_channels=HIDDEN_SIZE
    )
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log_message(f"Model initialized with {num_params:,} trainable parameters")
    print()
    
    # Initialize trainer
    trainer = ChessTrainer(model, learning_rate=LEARNING_RATE, device=device)
    
    # Initialize parallel game player
    game_player = ParallelGamePlayer(
        model=model,
        num_games=NUM_PARALLEL_GAMES,
        device=device,
        use_knowledge=USE_CHESS_KNOWLEDGE
    )
    
    # Training configuration
    log_message("Training Configuration:")
    log_message(f"  - Number of parallel games: {NUM_PARALLEL_GAMES}")
    log_message(f"  - Number of epochs: {NUM_EPOCHS}")
    log_message(f"  - Lookahead moves: {LOOKAHEAD_MOVES}")
    log_message(f"  - Learning rate: {LEARNING_RATE}")
    log_message(f"  - Save frequency: every {SAVE_FREQUENCY} epochs")
    log_message(f"  - Chess knowledge enabled: {USE_CHESS_KNOWLEDGE}")
    if USE_CHESS_KNOWLEDGE:
        log_message("    • Opening book will be used for first moves")
        log_message("    • Endgame knowledge will assist in endgame positions")
    print()
    
    # Check if checkpoint exists
    latest_checkpoint = os.path.join(CHECKPOINT_DIR, "latest_checkpoint.pt")
    start_epoch = 0
    
    if os.path.exists(latest_checkpoint):
        response = input("Found existing checkpoint. Load it? (y/n): ").strip().lower()
        if response == 'y':
            checkpoint = trainer.load_checkpoint(latest_checkpoint)
            start_epoch = checkpoint.get('epoch', 0) + 1
            log_message(f"Resumed training from epoch {start_epoch}")
            print()
    
    # Training loop
    log_message("Starting training...")
    log_message("=" * 80)
    print()
    
    try:
        for epoch in range(start_epoch, NUM_EPOCHS):
            epoch_start_time = time.time()
            
            log_message(f"EPOCH {epoch + 1}/{NUM_EPOCHS}")
            log_message("-" * 80)
            
            # Phase 1: Play games
            log_message("Phase 1: Playing games against Stockfish...")
            games_start_time = time.time()
            
            experiences = game_player.play_games()
            
            games_duration = time.time() - games_start_time
            log_message(f"Games completed in {games_duration:.2f} seconds")
            
            # Collect statistics
            num_games = len(experiences)
            num_moves = sum(len(exp.states) for exp in experiences)
            
            # Count wins, losses, draws
            wins = 0
            losses = 0
            draws = 0
            
            for exp in experiences:
                if exp.result == "1/2-1/2":
                    draws += 1
                elif exp.result == "1-0":
                    # White won
                    if exp.ai_plays_white:
                        wins += 1
                    else:
                        losses += 1
                elif exp.result == "0-1":
                    # Black won
                    if exp.ai_plays_white:
                        losses += 1
                    else:
                        wins += 1
            
            # Calculate win rate
            win_rate = (100.0 * wins / num_games) if num_games > 0 else 0.0
            
            log_message(f"  Games played: {num_games}")
            log_message(f"  Total moves: {num_moves}")
            log_message(f"  Results - Wins: {wins}, Draws: {draws}, Losses: {losses}")
            log_message(f"  Win Rate: {win_rate:.1f}%")
            print()
            
            # Phase 2: Train model
            log_message("Phase 2: Training neural network...")
            train_start_time = time.time()
            
            training_data = game_player.collect_training_data(experiences)
            
            if len(training_data['states']) > 0:
                loss_stats = trainer.train_epoch(training_data)
                
                # Record game results
                trainer.record_game_results(wins, draws, losses)
                
                train_duration = time.time() - train_start_time
                log_message(f"Training completed in {train_duration:.2f} seconds")
                log_message(f"  Policy Loss: {loss_stats['policy_loss']:.6f}")
                log_message(f"  Value Loss: {loss_stats['value_loss']:.6f}")
                log_message(f"  Total Loss: {loss_stats['total_loss']:.6f}")
            else:
                log_message("No training data collected (all games invalid)")
            
            print()
            
            # Phase 3: Save checkpoint and generate charts
            if (epoch + 1) % SAVE_FREQUENCY == 0:
                log_message("Phase 3: Saving checkpoint and generating charts...")
                
                checkpoint_path = os.path.join(
                    CHECKPOINT_DIR,
                    f"checkpoint_epoch_{epoch + 1}.pt"
                )
                
                trainer.save_checkpoint(
                    checkpoint_path,
                    epoch,
                    additional_info={
                        'games_played': num_games,
                        'wins': wins,
                        'draws': draws,
                        'losses': losses,
                        'win_rate': win_rate
                    }
                )
                
                # Also save as latest
                trainer.save_checkpoint(latest_checkpoint, epoch)
                
                # Generate training charts
                log_message("Generating training charts...")
                training_stats = trainer.get_loss_history()
                plot_training_progress(training_stats)
                print()
            
            # Epoch summary
            epoch_duration = time.time() - epoch_start_time
            log_message(f"Epoch {epoch + 1} completed in {epoch_duration:.2f} seconds")
            log_message("=" * 80)
            print()
            
            # Check if target win rate achieved
            if win_rate >= TARGET_WIN_RATE:
                log_message("=" * 80)
                log_message(f"🎉 TARGET ACHIEVED! Win rate: {win_rate:.1f}%")
                log_message("=" * 80)
                
                # Save final checkpoint
                final_checkpoint_path = os.path.join(CHECKPOINT_DIR, "final_model_100percent.pt")
                trainer.save_checkpoint(
                    final_checkpoint_path,
                    epoch,
                    additional_info={
                        'games_played': num_games,
                        'wins': wins,
                        'draws': draws,
                        'losses': losses,
                        'win_rate': win_rate,
                        'target_achieved': True
                    }
                )
                
                # Generate final charts
                log_message("Generating final training charts...")
                training_stats = trainer.get_loss_history()
                plot_training_progress(training_stats)
                
                log_message(f"Training completed successfully!")
                log_message(f"Final model saved to: {final_checkpoint_path}")
                break
    
    except KeyboardInterrupt:
        log_message("\nTraining interrupted by user")
        log_message("Saving current state...")
        trainer.save_checkpoint(
            os.path.join(CHECKPOINT_DIR, "interrupted_checkpoint.pt"),
            epoch
        )
        log_message("Checkpoint saved. Training can be resumed later.")
    
    except Exception as e:
        log_message(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
        log_message("Saving emergency checkpoint...")
        trainer.save_checkpoint(
            os.path.join(CHECKPOINT_DIR, "emergency_checkpoint.pt"),
            epoch
        )
    
    finally:
        log_message("\nTraining session ended")
        log_message(f"Total epochs completed: {epoch - start_epoch + 1}")
        log_message(f"Checkpoints saved in: {CHECKPOINT_DIR}")


if __name__ == "__main__":
    main()
