"""Test and play against the trained chess engine."""

import os
import torch
import chess
import chess.svg
from model import ChessNet
from chess_env import ChessEnvironment
from stockfish_opponent import StockfishOpponent
from config import USE_CUDA, CHECKPOINT_DIR, HIDDEN_SIZE, NUM_RESIDUAL_BLOCKS


class ChessEnginePlayer:
    """Wrapper for the trained chess engine."""
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.eval()
    
    def get_move(self, env, temperature=1.0):
        """
        Get a move from the engine.
        
        Args:
            env: ChessEnvironment object
            temperature: Temperature for move selection (higher = more random)
            
        Returns:
            chess.Move object
        """
        state = env.encode_board().to(self.device)
        legal_moves_mask = env.get_legal_moves_mask().to(self.device)
        
        with torch.no_grad():
            policy, value = self.model(state.unsqueeze(0))
            policy = policy.squeeze(0)
            
            # Mask illegal moves
            policy = policy.masked_fill(~legal_moves_mask, float('-inf'))
            
            # Apply temperature
            policy = policy / temperature
            
            # Get move probabilities
            move_probs = torch.softmax(policy, dim=0)
            
            # Sample move
            move_idx = torch.multinomial(move_probs, 1).item()
        
        # Convert index to move
        move = env.index_to_move(move_idx)
        
        # Fallback to random legal move if invalid
        if move is None or move not in env.board.legal_moves:
            legal_moves = list(env.board.legal_moves)
            import random
            move = random.choice(legal_moves)
        
        return move, value.item()


def load_latest_checkpoint():
    """Load the latest checkpoint."""
    device = torch.device("cuda" if USE_CUDA and torch.cuda.is_available() else "cpu")
    
    # Initialize model
    model = ChessNet(
        num_residual_blocks=NUM_RESIDUAL_BLOCKS,
        num_channels=HIDDEN_SIZE
    )
    model.to(device)
    
    # Find latest checkpoint
    checkpoint_files = [
        f for f in os.listdir(CHECKPOINT_DIR) 
        if f.endswith('.pt') and 'checkpoint' in f
    ]
    
    if not checkpoint_files:
        print("No checkpoints found!")
        return None, None
    
    # Load latest or specific checkpoint
    if 'latest_checkpoint.pt' in checkpoint_files:
        checkpoint_path = os.path.join(CHECKPOINT_DIR, 'latest_checkpoint.pt')
    else:
        # Get the most recent checkpoint by epoch number
        epoch_checkpoints = [f for f in checkpoint_files if 'epoch' in f]
        if epoch_checkpoints:
            latest = sorted(epoch_checkpoints, 
                          key=lambda x: int(x.split('_')[-1].replace('.pt', '')))[-1]
            checkpoint_path = os.path.join(CHECKPOINT_DIR, latest)
        else:
            checkpoint_path = os.path.join(CHECKPOINT_DIR, checkpoint_files[0])
    
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    epoch = checkpoint.get('epoch', 'unknown')
    print(f"Loaded model from epoch {epoch}")
    
    return model, device


def play_against_human():
    """Play a game: Human vs AI."""
    print("\n" + "="*80)
    print("PLAY AGAINST THE CHESS ENGINE")
    print("="*80 + "\n")
    
    model, device = load_latest_checkpoint()
    if model is None:
        return
    
    engine = ChessEnginePlayer(model, device)
    env = ChessEnvironment()
    
    # Choose color
    while True:
        color = input("Do you want to play as White or Black? (w/b): ").strip().lower()
        if color in ['w', 'white']:
            human_is_white = True
            break
        elif color in ['b', 'black']:
            human_is_white = False
            break
        print("Invalid input. Please enter 'w' or 'b'.")
    
    print(f"\nYou are playing as {'White' if human_is_white else 'Black'}")
    print("Enter moves in UCI format (e.g., e2e4, g1f3)")
    print("Type 'quit' to exit\n")
    
    move_count = 0
    
    while not env.board.is_game_over():
        env.render()
        
        is_human_turn = (env.board.turn == chess.WHITE) == human_is_white
        
        if is_human_turn:
            # Human's turn
            while True:
                move_str = input("Your move: ").strip().lower()
                
                if move_str == 'quit':
                    print("Game ended by user.")
                    return
                
                try:
                    move = chess.Move.from_uci(move_str)
                    if move in env.board.legal_moves:
                        break
                    else:
                        print("Illegal move! Try again.")
                except ValueError:
                    print("Invalid move format! Use UCI notation (e.g., e2e4)")
            
            env.step(move)
        else:
            # AI's turn
            print("AI is thinking...")
            move, value = engine.get_move(env, temperature=0.5)
            print(f"AI plays: {move.uci()} (Position value: {value:.3f})")
            env.step(move)
        
        move_count += 1
    
    # Game over
    env.render()
    result = env.board.result()
    print(f"\nGame Over! Result: {result}")
    
    if result == "1-0":
        winner = "White wins!"
    elif result == "0-1":
        winner = "Black wins!"
    else:
        winner = "Draw!"
    
    print(winner)


def test_against_stockfish(num_games=10, stockfish_level=5):
    """Test the engine against Stockfish."""
    print("\n" + "="*80)
    print(f"TESTING ENGINE AGAINST STOCKFISH (Level {stockfish_level})")
    print("="*80 + "\n")
    
    model, device = load_latest_checkpoint()
    if model is None:
        return
    
    engine = ChessEnginePlayer(model, device)
    stockfish = StockfishOpponent(skill_level=stockfish_level)
    
    try:
        stockfish.start()
        
        results = {"wins": 0, "losses": 0, "draws": 0}
        
        for game_num in range(num_games):
            env = ChessEnvironment()
            ai_is_white = (game_num % 2 == 0)
            
            print(f"Game {game_num + 1}/{num_games} - AI plays as {'White' if ai_is_white else 'Black'}")
            
            move_count = 0
            max_moves = 200
            
            while not env.board.is_game_over() and move_count < max_moves:
                is_ai_turn = (env.board.turn == chess.WHITE) == ai_is_white
                
                if is_ai_turn:
                    move, value = engine.get_move(env, temperature=0.3)
                else:
                    move = stockfish.get_move(env.board)
                
                env.step(move)
                move_count += 1
            
            # Determine result
            if env.board.is_game_over():
                result = env.board.result()
            else:
                result = "1/2-1/2"
            
            # Count result
            if result == "1-0":
                if ai_is_white:
                    results["wins"] += 1
                    print(f"  Result: AI wins!\n")
                else:
                    results["losses"] += 1
                    print(f"  Result: AI loses.\n")
            elif result == "0-1":
                if ai_is_white:
                    results["losses"] += 1
                    print(f"  Result: AI loses.\n")
                else:
                    results["wins"] += 1
                    print(f"  Result: AI wins!\n")
            else:
                results["draws"] += 1
                print(f"  Result: Draw.\n")
        
        # Print summary
        print("="*80)
        print("TEST RESULTS")
        print("="*80)
        print(f"Games played: {num_games}")
        print(f"Wins:   {results['wins']} ({results['wins']/num_games*100:.1f}%)")
        print(f"Losses: {results['losses']} ({results['losses']/num_games*100:.1f}%)")
        print(f"Draws:  {results['draws']} ({results['draws']/num_games*100:.1f}%)")
        
        win_rate = (results['wins'] + 0.5 * results['draws']) / num_games * 100
        print(f"\nWin rate (counting draws as 0.5): {win_rate:.1f}%")
        
    finally:
        stockfish.close()


def watch_ai_vs_stockfish():
    """Watch a single game: AI vs Stockfish."""
    print("\n" + "="*80)
    print("WATCH: AI vs STOCKFISH")
    print("="*80 + "\n")
    
    model, device = load_latest_checkpoint()
    if model is None:
        return
    
    engine = ChessEnginePlayer(model, device)
    stockfish = StockfishOpponent(skill_level=5)
    
    try:
        stockfish.start()
        env = ChessEnvironment()
        
        # AI plays white
        ai_is_white = True
        print(f"AI plays as White, Stockfish plays as Black\n")
        
        move_count = 0
        max_moves = 200
        
        while not env.board.is_game_over() and move_count < max_moves:
            env.render()
            
            is_ai_turn = (env.board.turn == chess.WHITE) == ai_is_white
            
            if is_ai_turn:
                print("AI is thinking...")
                move, value = engine.get_move(env, temperature=0.5)
                print(f"AI plays: {move.uci()} (Value: {value:.3f})\n")
            else:
                print("Stockfish is thinking...")
                move = stockfish.get_move(env.board)
                print(f"Stockfish plays: {move.uci()}\n")
            
            env.step(move)
            move_count += 1
            
            input("Press Enter for next move...")
        
        env.render()
        result = env.board.result()
        print(f"\nGame Over! Result: {result}")
        
    finally:
        stockfish.close()


def analyze_position():
    """Analyze a specific position."""
    print("\n" + "="*80)
    print("POSITION ANALYSIS")
    print("="*80 + "\n")
    
    model, device = load_latest_checkpoint()
    if model is None:
        return
    
    engine = ChessEnginePlayer(model, device)
    env = ChessEnvironment()
    
    print("Enter FEN notation (or press Enter for starting position):")
    fen = input().strip()
    
    if fen:
        try:
            env.board = chess.Board(fen)
        except ValueError:
            print("Invalid FEN! Using starting position.")
    
    env.render()
    
    # Get top moves
    state = env.encode_board().to(device)
    legal_moves_mask = env.get_legal_moves_mask().to(device)
    
    with torch.no_grad():
        policy, value = model(state.unsqueeze(0))
        policy = policy.squeeze(0)
        
        # Mask illegal moves
        policy = policy.masked_fill(~legal_moves_mask, float('-inf'))
        move_probs = torch.softmax(policy, dim=0)
    
    print(f"\nPosition evaluation: {value.item():.3f}")
    print(f"(Positive = good for {'White' if env.board.turn == chess.WHITE else 'Black'})\n")
    
    # Get top 5 moves
    legal_moves = list(env.board.legal_moves)
    move_scores = []
    
    for move in legal_moves:
        move_idx = env.move_to_index(move)
        if 0 <= move_idx < 4672:
            prob = move_probs[move_idx].item()
            move_scores.append((move, prob))
    
    move_scores.sort(key=lambda x: x[1], reverse=True)
    
    print("Top 5 moves:")
    for i, (move, prob) in enumerate(move_scores[:5], 1):
        print(f"{i}. {move.uci()}: {prob*100:.2f}%")


def main():
    """Main menu for testing the engine."""
    while True:
        print("\n" + "="*80)
        print("CHESS ENGINE TESTING MENU")
        print("="*80)
        print("1. Play against the AI")
        print("2. Test AI vs Stockfish (multiple games)")
        print("3. Watch AI vs Stockfish (single game)")
        print("4. Analyze a position")
        print("5. Exit")
        print("="*80)
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == "1":
            play_against_human()
        elif choice == "2":
            num_games = input("Number of games (default 10): ").strip()
            num_games = int(num_games) if num_games.isdigit() else 10
            
            level = input("Stockfish level 0-20 (default 5): ").strip()
            level = int(level) if level.isdigit() else 5
            
            test_against_stockfish(num_games, level)
        elif choice == "3":
            watch_ai_vs_stockfish()
        elif choice == "4":
            analyze_position()
        elif choice == "5":
            print("Goodbye!")
            break
        else:
            print("Invalid choice! Please enter 1-5.")


if __name__ == "__main__":
    main()
