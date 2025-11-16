"""Test script to verify opening book integration."""

import torch
from model import ChessNet
from parallel_player import ParallelGamePlayer
from config import HIDDEN_SIZE, NUM_RESIDUAL_BLOCKS, USE_CUDA

def test_opening_book():
    """Test that the opening book is being used."""
    print("=" * 80)
    print("TESTING OPENING BOOK INTEGRATION")
    print("=" * 80)
    print()
    
    # Initialize model
    device = torch.device("cuda" if USE_CUDA and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = ChessNet(
        num_channels=HIDDEN_SIZE,
        num_residual_blocks=NUM_RESIDUAL_BLOCKS
    )
    model.to(device)
    model.eval()
    
    print("Model initialized")
    print()
    
    # Test with knowledge enabled
    print("Testing with opening book ENABLED...")
    game_player = ParallelGamePlayer(
        model=model,
        num_games=2,
        device=device,
        use_knowledge=True
    )
    
    print("Playing 2 test games...")
    experiences = game_player.play_games()
    print(f"Completed! Generated {len(experiences)} game experiences")
    print()
    
    # Check games.log for book moves
    print("Check games.log to see if opening book moves were used!")
    print("Look for moves marked as 'AI-opening_book' in the log")
    print()
    
    print("=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    test_opening_book()
