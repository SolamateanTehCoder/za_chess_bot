"""
Tournament framework for evaluating chess bot strength.
Supports round-robin tournaments, Elo rating calculation, and PGN export.
Essential for benchmarking and WCCC qualification.
"""

import chess
import chess.pgn
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json
import statistics
from enum import Enum


class GameResult(Enum):
    """Game result types."""
    WHITE_WIN = "1-0"
    BLACK_WIN = "0-1"
    DRAW = "1/2-1/2"


class TournamentGame:
    """Single game in tournament."""
    
    def __init__(self, white_player: str, black_player: str, round_num: int = 1):
        """
        Initialize tournament game.
        
        Args:
            white_player: Name of white player
            black_player: Name of black player
            round_num: Round number
        """
        self.white_player = white_player
        self.black_player = black_player
        self.round_num = round_num
        self.result: Optional[GameResult] = None
        self.board = chess.Board()
        self.moves: List[chess.Move] = []
        self.start_time = None
        self.end_time = None
        self.white_time_used = 0
        self.black_time_used = 0
        self.opening_eco = None
        self.opening_name = None
    
    def add_move(self, move: chess.Move, move_time: float = 0):
        """
        Record a move in the game.
        
        Args:
            move: Chess move
            move_time: Time used for move (seconds)
        """
        if self.board.turn == chess.WHITE:
            self.white_time_used += move_time
        else:
            self.black_time_used += move_time
        
        self.moves.append(move)
        self.board.push(move)
    
    def end_game(self, result: GameResult):
        """
        End the game with result.
        
        Args:
            result: Game result
        """
        self.result = result
        self.end_time = datetime.now()
    
    def to_pgn(self) -> str:
        """
        Convert game to PGN format.
        
        Returns:
            PGN string
        """
        game = chess.pgn.Game()
        game.headers["Event"] = "Tournament"
        game.headers["White"] = self.white_player
        game.headers["Black"] = self.black_player
        game.headers["Result"] = self.result.value if self.result else "*"
        game.headers["Round"] = str(self.round_num)
        
        if self.opening_eco:
            game.headers["ECO"] = self.opening_eco
        if self.opening_name:
            game.headers["Opening"] = self.opening_name
        
        node = game
        for move in self.moves:
            node = node.add_variation(move)
        
        return str(game)
    
    def get_pgn_object(self) -> chess.pgn.Game:
        """Get PGN game object."""
        game = chess.pgn.Game()
        game.headers["Event"] = "Tournament"
        game.headers["White"] = self.white_player
        game.headers["Black"] = self.black_player
        game.headers["Result"] = self.result.value if self.result else "*"
        game.headers["Round"] = str(self.round_num)
        game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
        
        if self.opening_eco:
            game.headers["ECO"] = self.opening_eco
        if self.opening_name:
            game.headers["Opening"] = self.opening_name
        
        node = game
        for move in self.moves:
            node = node.add_variation(move)
        
        return game


class EloRating:
    """Elo rating calculator."""
    
    K_FACTOR = 32  # Standard K-factor
    
    @staticmethod
    def expected_score(rating1: float, rating2: float) -> float:
        """
        Calculate expected score for player 1.
        
        Args:
            rating1: Rating of player 1
            rating2: Rating of player 2
            
        Returns:
            Expected score (0.0 to 1.0)
        """
        return 1.0 / (1.0 + 10.0 ** ((rating2 - rating1) / 400.0))
    
    @staticmethod
    def update_rating(current_rating: float, expected: float, actual: float, k_factor: int = 32) -> float:
        """
        Update rating based on game result.
        
        Args:
            current_rating: Current rating
            expected: Expected score
            actual: Actual score (1.0=win, 0.5=draw, 0.0=loss)
            k_factor: K-factor (higher = more volatile)
            
        Returns:
            Updated rating
        """
        return current_rating + k_factor * (actual - expected)


class Tournament:
    """Tournament manager."""
    
    def __init__(self, name: str, time_control: str = "60+0"):
        """
        Initialize tournament.
        
        Args:
            name: Tournament name
            time_control: Time control string (e.g., "60+0")
        """
        self.name = name
        self.time_control = time_control
        self.games: List[TournamentGame] = []
        self.player_ratings: Dict[str, float] = {}
        self.player_stats: Dict[str, Dict] = {}
        self.start_time = None
        self.end_time = None
    
    def add_player(self, name: str, initial_rating: float = 1600):
        """
        Add player to tournament.
        
        Args:
            name: Player name
            initial_rating: Initial Elo rating
        """
        self.player_ratings[name] = initial_rating
        self.player_stats[name] = {
            "wins": 0,
            "draws": 0,
            "losses": 0,
            "games": 0,
            "white_score": 0,
            "white_games": 0,
            "black_score": 0,
            "black_games": 0
        }
    
    def record_game(self, white_player: str, black_player: str, result: GameResult, 
                   round_num: int = 1, moves: List[chess.Move] = None):
        """
        Record game result.
        
        Args:
            white_player: White player name
            black_player: Black player name
            result: Game result
            round_num: Round number
            moves: List of moves played
        """
        game = TournamentGame(white_player, black_player, round_num)
        
        if moves:
            for move in moves:
                game.add_move(move)
        
        game.end_game(result)
        self.games.append(game)
        
        # Update stats
        if result == GameResult.WHITE_WIN:
            self.player_stats[white_player]["wins"] += 1
            self.player_stats[black_player]["losses"] += 1
            white_score, black_score = 1.0, 0.0
        elif result == GameResult.BLACK_WIN:
            self.player_stats[white_player]["losses"] += 1
            self.player_stats[black_player]["wins"] += 1
            white_score, black_score = 0.0, 1.0
        else:  # Draw
            self.player_stats[white_player]["draws"] += 1
            self.player_stats[black_player]["draws"] += 1
            white_score, black_score = 0.5, 0.5
        
        # Update game counts
        self.player_stats[white_player]["games"] += 1
        self.player_stats[white_player]["white_score"] += white_score
        self.player_stats[white_player]["white_games"] += 1
        
        self.player_stats[black_player]["games"] += 1
        self.player_stats[black_player]["black_score"] += black_score
        self.player_stats[black_player]["black_games"] += 1
        
        # Update Elo ratings
        white_expected = EloRating.expected_score(
            self.player_ratings[white_player],
            self.player_ratings[black_player]
        )
        black_expected = 1.0 - white_expected
        
        self.player_ratings[white_player] = EloRating.update_rating(
            self.player_ratings[white_player],
            white_expected,
            white_score
        )
        self.player_ratings[black_player] = EloRating.update_rating(
            self.player_ratings[black_player],
            black_expected,
            black_score
        )
    
    def get_standings(self) -> List[Tuple[str, int, int, int, float, float]]:
        """
        Get tournament standings.
        
        Returns:
            List of (player_name, wins, draws, losses, score, rating)
        """
        standings = []
        
        for player in sorted(self.player_stats.keys()):
            stats = self.player_stats[player]
            score = stats["wins"] + stats["draws"] * 0.5
            rating = self.player_ratings[player]
            
            standings.append((
                player,
                stats["wins"],
                stats["draws"],
                stats["losses"],
                score,
                rating
            ))
        
        # Sort by score descending
        standings.sort(key=lambda x: x[4], reverse=True)
        
        return standings
    
    def print_standings(self):
        """Print tournament standings."""
        print(f"\n=== {self.name} ===")
        print(f"Time Control: {self.time_control}\n")
        
        standings = self.get_standings()
        
        print(f"{'Rank':<5} {'Player':<20} {'W':<3} {'D':<3} {'L':<3} {'Score':<8} {'Rating':<8}")
        print("-" * 60)
        
        for rank, (player, wins, draws, losses, score, rating) in enumerate(standings, 1):
            print(f"{rank:<5} {player:<20} {wins:<3} {draws:<3} {losses:<3} {score:<8.1f} {rating:<8.0f}")
    
    def export_pgn(self, filepath: str):
        """
        Export all games to PGN file.
        
        Args:
            filepath: Path to save PGN file
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            for game in self.games:
                pgn = game.get_pgn_object()
                f.write(str(pgn))
                f.write("\n\n")
        
        print(f"[INFO] Exported {len(self.games)} games to {filepath}")
    
    def export_json(self, filepath: str):
        """
        Export tournament results to JSON.
        
        Args:
            filepath: Path to save JSON file
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "name": self.name,
            "time_control": self.time_control,
            "total_games": len(self.games),
            "standings": [],
            "ratings": {}
        }
        
        standings = self.get_standings()
        for rank, (player, wins, draws, losses, score, rating) in enumerate(standings, 1):
            data["standings"].append({
                "rank": rank,
                "player": player,
                "wins": wins,
                "draws": draws,
                "losses": losses,
                "score": score,
                "rating": rating
            })
            data["ratings"][player] = rating
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"[INFO] Exported results to {filepath}")


class TournamentRunner:
    """Run complete tournament (round-robin, Swiss, etc.)."""
    
    @staticmethod
    def round_robin(players: List[str], name: str = "Tournament",
                   time_control: str = "60+0") -> Tournament:
        """
        Run round-robin tournament.
        
        Args:
            players: List of player names
            name: Tournament name
            time_control: Time control
            
        Returns:
            Completed tournament
        """
        tournament = Tournament(name, time_control)
        
        # Add players
        for player in players:
            tournament.add_player(player)
        
        print(f"\n[INFO] Starting {name} Round-Robin Tournament")
        print(f"[INFO] Players: {players}")
        print(f"[INFO] Time Control: {time_control}\n")
        
        # Generate pairings
        pairings = []
        n = len(players)
        for round_num in range(n - 1):
            round_pairings = []
            for i in range(n // 2):
                white_idx = (round_num + i) % (n - 1)
                black_idx = (n - 1 - i) % (n - 1) if i != 0 else n - 1
                
                white = players[white_idx]
                black = players[black_idx]
                round_pairings.append((white, black, round_num + 1))
            
            pairings.extend(round_pairings)
        
        print(f"[INFO] Total games: {len(pairings)}")
        print(f"[INFO] Waiting for game results...\n")
        
        # Note: Games would be played by the actual engines
        # This is the framework; actual play happens elsewhere
        
        return tournament
