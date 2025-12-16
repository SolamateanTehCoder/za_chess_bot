"""
Train against Stockfish with "Pain" Mechanism.

This script implements a high-intensity training mode where the bot plays directly 
against Stockfish. Losses incur a massive negative reward ("pain") to force the 
bot to learn avoidance of losing lines.

Features:
- Direct play vs Stockfish
- "Pain" Reinforcement: -10.0 for Loss, +10.0 for Win
- Speed Optimization: Fast Stockfish (0.05s) + No internal bot verification
"""

import chess
import chess.engine
import torch
import numpy as np
import json
import time as time_module
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime

# Import existing components
from hybrid_player import HybridChessPlayer
from chess_models import ChessNetV2
from advanced_trainer import AdvancedTrainer

class StockfishOpponent:
    """Fast Stockfish opponent."""
    
    def __init__(self, depth: int = 10, time_limit: float = 0.05):
        self.depth = depth
        self.time_limit = time_limit
        self.engine = None
        self._init_engine()
        
    def _init_engine(self):
        """Initialize Stockfish."""
        paths = [
            r"C:\stockfish\stockfish-windows-x86-64-avx2.exe",
            "stockfish"
        ]
        
        for path in paths:
            try:
                self.engine = chess.engine.SimpleEngine.popen_uci(path)
                # Configure for speed
                self.engine.configure({"Threads": 1, "Hash": 16})
                print(f"[OPPONENT] Stockfish active at {path} (Time: {self.time_limit}s)")
                return
            except:
                continue
                
        print("[ERROR] Could not start Stockfish opponent!")
        
    def get_move(self, board: chess.Board) -> chess.Move:
        """Get fast move from Stockfish."""
        if not self.engine:
            return list(board.legal_moves)[0]
            
        try:
            limit = chess.engine.Limit(time=self.time_limit, depth=self.depth)
            result = self.engine.play(board, limit)
            return result.move
        except:
            return list(board.legal_moves)[0]
            
    def close(self):
        if self.engine:
            self.engine.quit()

class PainTrainer:
    """Trainer that applies high penalties for losses."""
    
    def __init__(self, device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = ChessNetV2().to(self.device)
        self.opponent = StockfishOpponent()
        
        # Load latest model if exists
        self._load_latest()
        
        # Training components
        self.trainer = AdvancedTrainer(self.model, self.device, learning_rate=0.0005)
        self.player = HybridChessPlayer(model=self.model, device=self.device)
        
    def _load_latest(self):
        """Load latest checkpoint."""
        checkpoint_dir = Path("checkpoints_strategy")
        if checkpoint_dir.exists():
            checkpoints = list(checkpoint_dir.glob("*.pt"))
            if checkpoints:
                latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
                try:
                    self.model.load_state_dict(torch.load(latest, map_location=self.device))
                    print(f"[INIT] Loaded model: {latest.name}")
                except:
                    print("[INIT] Starting with fresh model")

    def play_pain_game(self, play_white: bool) -> Dict:
        """
        Play a game and return training data with PAIN rewards.
        """
        board = chess.Board()
        game_data = [] # List of (board_tensor, move_idx) for OUR moves
        
        moves_played = 0
        
        while not board.is_game_over():
            is_turn = (board.turn == chess.WHITE) == play_white
            
            if is_turn:
                # Bot move - pure Neural Network (fastest)
                # We interpret HybridPlayer logic here locally for speed and data capture
                encoded = self.player.encode_board(board)
                
                with torch.no_grad():
                    policy, _ = self.model(encoded.unsqueeze(0))
                    
                # Select move
                uci_move = self.player.get_move_from_policy(board, policy)
                move = chess.Move.from_uci(uci_move) if uci_move else None
                
                if move is None:
                    move = list(board.legal_moves)[0] # Fallback
                
                # Record state for training
                game_data.append({
                    "tensor": encoded,
                    "move": move,
                    "turn": board.turn
                })
                
                board.push(move)
            else:
                # Stockfish move
                move = self.opponent.get_move(board)
                board.push(move)
                
            moves_played += 1
            if moves_played > 200: # preventing infinite games
                break
        
        # Determine Result and PAIN
        result = board.result()
        if "1-0" in result:
            outcome = 1.0 if play_white else 0.0
        elif "0-1" in result:
            outcome = 0.0 if play_white else 1.0
        else:
            outcome = 0.5
            
        # Reward Shaping (The "Pain" Logic)
        if outcome == 1.0:
            reward = 10.0  # Big reward for beating Stockfish
            log_msg = "WIN! (+10.0)"
        elif outcome == 0.0:
            reward = -10.0 # PAIN!
            log_msg = "LOSS (-10.0 PAIN)"
        else:
            reward = 0.0
            log_msg = "DRAW (0.0)"
            
        print(f"[GAME] Result: {result} ({moves_played} moves) -> {log_msg}")
        
        return {
            "data": game_data,
            "reward": reward,
            "outcome": outcome
        }

    def train_cycle(self, games=10, epochs=5):
        """Run a training cycle."""
        print(f"\n[CYCLE] Playing {games} vs Stockfish...")
        
        all_experiences = []
        
        # Play Games
        for i in range(games):
            # Alternate colors
            play_white = (i % 2 == 0)
            result = self.play_pain_game(play_white)
            
            # Convert game data to training samples
            # We assign the FINAL reward to ALL moves (simple Monte Carlo)
            final_reward = result['reward']
            outcome = result['outcome']
            
            for move_data in result['data']:
                tensor = move_data['tensor']
                move = move_data['move']
                
                # Find move index
                board_dummy = chess.Board() # Need board context for legal moves?
                # Actually we can just use the raw move index calculation
                move_idx = move.from_square * 64 + move.to_square
                
                all_experiences.append({
                    "position": tensor,
                    "move": torch.tensor([move_idx], dtype=torch.long),
                    "reward": torch.tensor([final_reward], dtype=torch.float32),
                    "outcome": torch.tensor([outcome], dtype=torch.float32)
                })
                
        print(f"[TRAIN] Collected {len(all_experiences)} samples. Training...")
        
        # Create Batch Loader
        class PainDataset(torch.utils.data.Dataset):
            def __init__(self, data): self.data = data
            def __len__(self): return len(self.data)
            def __getitem__(self, idx): return self.data[idx]
            
        loader = torch.utils.data.DataLoader(
            PainDataset(all_experiences), 
            batch_size=32, 
            shuffle=True
        )
        
        # Train
        self.trainer.train_epoch(loader, epochs=epochs)
        
        # Save
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = f"checkpoints_strategy/pain_model_{timestamp}.pt"
        torch.save(self.model.state_dict(), path)
        print(f"[SAVE] Checkpoint: {path}")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()
    
    print("="*60)
    print("PAIN TRAINING: VS STOCKFISH")
    print("Penalty: -10.0 | Reward: +10.0")
    print("="*60)
    
    trainer = PainTrainer()
    try:
        while True:
            trainer.train_cycle(args.games, args.epochs)
    except KeyboardInterrupt:
        print("\n[STOP] Training stopped.")
        trainer.opponent.close()

if __name__ == "__main__":
    main()
