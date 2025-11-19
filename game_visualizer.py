"""
Real-time GUI to visualize all 28 chess games at once.
Shows board positions, timers with color feedback (green=reward, red=pain), and move tracking.
"""

import tkinter as tk
from tkinter import Canvas, Label, Frame
import chess
import chess.svg
import time
from datetime import datetime
from typing import Dict, List, Optional
from threading import Thread, Lock
import queue


class GameVisualizerGUI:
    """Multi-board chess game visualizer with 28 games displayed simultaneously."""
    
    def __init__(self, num_games: int = 28, num_cols: int = 7):
        """
        Initialize the game visualizer GUI.
        
        Args:
            num_games: Total number of games to display (default 28)
            num_cols: Number of columns in grid layout
        """
        self.num_games = num_games
        self.num_cols = num_cols
        self.num_rows = (num_games + num_cols - 1) // num_cols
        
        # Board size (smaller for 28 games on screen)
        self.board_size = 150  # pixels
        self.tile_size = self.board_size // 8
        
        # Game state storage
        self.games = {}
        self.game_states = {}
        self.timers = {}
        self.accuracies = {}
        self.reward_flashes = {}  # For green/red timer flashes
        
        # Threading
        self.update_queue = queue.Queue()
        self.running = True
        self.lock = Lock()
        
        # Window setup
        self.root = tk.Tk()
        self.root.title("Chess Bot Self-Play Visualizer - 28 Games")
        self.root.geometry(f"{self.board_size * self.num_cols + 50}x{self.board_size * self.num_rows + 150}")
        
        # Create main frame
        self.main_frame = Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create status bar at top
        self.status_frame = Frame(self.root)
        self.status_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.status_label = Label(self.status_frame, text="Initializing...", font=("Arial", 10))
        self.status_label.pack(side=tk.LEFT)
        
        self.epoch_label = Label(self.status_frame, text="Epoch: -", font=("Arial", 10))
        self.epoch_label.pack(side=tk.RIGHT)
        
        # Create game frames
        self.game_frames = {}
        self.board_canvases = {}
        self.timer_labels = {}
        self.accuracy_labels = {}
        self.result_labels = {}
        
        for game_id in range(num_games):
            row = game_id // self.num_cols
            col = game_id % self.num_cols
            
            # Game frame
            frame = Frame(self.main_frame, relief=tk.RAISED, borderwidth=1)
            frame.grid(row=row, column=col, padx=2, pady=2, sticky="nsew")
            
            # Game ID label
            id_label = Label(frame, text=f"Game {game_id + 1}", font=("Arial", 8, "bold"))
            id_label.pack()
            
            # Canvas for board (minimal display)
            canvas = Canvas(frame, width=self.board_size, height=self.board_size, bg="white")
            canvas.pack()
            self.board_canvases[game_id] = canvas
            
            # Timer label (with color feedback)
            timer_label = Label(frame, text="00:00 | 00:00", font=("Arial", 7), fg="black", bg="white")
            timer_label.pack()
            self.timer_labels[game_id] = timer_label
            
            # Accuracy label
            accuracy_label = Label(frame, text="W: --% | B: --%", font=("Arial", 7))
            accuracy_label.pack()
            self.accuracy_labels[game_id] = accuracy_label
            
            # Result label
            result_label = Label(frame, text="", font=("Arial", 7, "bold"))
            result_label.pack()
            self.result_labels[game_id] = result_label
            
            self.game_frames[game_id] = frame
            
            # Initialize game state
            self.games[game_id] = {
                'board': chess.Board(),
                'white_time': 60.0,
                'black_time': 60.0,
                'moves': 0,
                'white_accuracy': 0.0,
                'black_accuracy': 0.0,
                'result': None,
                'last_reward': None,  # Track last reward for color feedback
                'last_reward_time': 0  # When the last reward/pain was applied
            }
    
    def update_game(self, game_id: int, board: chess.Board, white_time: float, black_time: float,
                   moves: int, white_accuracy: float = 0.0, black_accuracy: float = 0.0,
                   reward: float = 0.0, result: Optional[str] = None):
        """
        Update a game's display state.
        
        Args:
            game_id: Game identifier (0-27)
            board: Current chess board
            white_time: Remaining time for white (seconds)
            black_time: Remaining time for black (seconds)
            moves: Number of moves played
            white_accuracy: White's average accuracy
            black_accuracy: Black's average accuracy
            reward: Last move's reward (positive=green, negative=red)
            result: Game result if finished
        """
        self.update_queue.put({
            'game_id': game_id,
            'board': board.copy() if board else None,
            'white_time': white_time,
            'black_time': black_time,
            'moves': moves,
            'white_accuracy': white_accuracy,
            'black_accuracy': black_accuracy,
            'reward': reward,
            'result': result
        })
    
    def _process_updates(self):
        """Process queued game updates from the main thread."""
        try:
            while True:
                update = self.update_queue.get_nowait()
                game_id = update['game_id']
                
                with self.lock:
                    game = self.games[game_id]
                    if update['board']:
                        game['board'] = update['board']
                    game['white_time'] = update['white_time']
                    game['black_time'] = update['black_time']
                    game['moves'] = update['moves']
                    game['white_accuracy'] = update['white_accuracy']
                    game['black_accuracy'] = update['black_accuracy']
                    game['result'] = update.get('result')
                    
                    # Track reward for color flash
                    if update.get('reward') is not None:
                        game['last_reward'] = update['reward']
                        game['last_reward_time'] = time.time()
        except queue.Empty:
            pass
    
    def _format_time(self, seconds: float) -> str:
        """Format seconds to MM:SS."""
        minutes = int(seconds) // 60
        secs = int(seconds) % 60
        return f"{minutes:02d}:{secs:02d}"
    
    def _draw_simple_board(self, canvas: Canvas, board: chess.Board):
        """Draw a simplified chess board with piece positions."""
        canvas.delete("all")
        
        # Draw checkerboard
        for row in range(8):
            for col in range(8):
                x1 = col * self.tile_size
                y1 = row * self.tile_size
                x2 = x1 + self.tile_size
                y2 = y1 + self.tile_size
                
                color = "white" if (row + col) % 2 == 0 else "gray"
                canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="black", width=0)
        
        # Draw pieces as text (simple representation)
        piece_symbols = {
            chess.PAWN: '♟',
            chess.KNIGHT: '♞',
            chess.BISHOP: '♝',
            chess.ROOK: '♜',
            chess.QUEEN: '♛',
            chess.KING: '♚'
        }
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                row = square // 8
                col = square % 8
                x = col * self.tile_size + self.tile_size // 2
                y = row * self.tile_size + self.tile_size // 2
                
                symbol = piece_symbols.get(piece.piece_type, '?')
                color = "white" if piece.color == chess.WHITE else "black"
                
                canvas.create_text(x, y, text=symbol, font=("Arial", 12, "bold"), fill=color)
    
    def _get_timer_color(self, game_id: int) -> str:
        """
        Get timer label color based on last reward.
        Green = received reward (PAIN)
        Red = received penalty (PAIN)
        White = neutral
        """
        game = self.games[game_id]
        
        if game['last_reward'] is None:
            return "black"
        
        # Color fade effect: bright for 0.5s, then fade
        time_since_reward = time.time() - game['last_reward_time']
        if time_since_reward > 0.5:
            game['last_reward'] = None  # Clear after fade
            return "black"
        
        if game['last_reward'] > 0:
            return "green"  # REWARD (positive)
        else:
            return "red"  # PAIN (negative)
    
    def refresh(self):
        """Refresh the display with current game states."""
        self._process_updates()
        
        with self.lock:
            for game_id in range(self.num_games):
                game = self.games[game_id]
                
                # Update board display
                self._draw_simple_board(self.board_canvases[game_id], game['board'])
                
                # Update timers with color feedback
                white_time_str = self._format_time(max(0, game['white_time']))
                black_time_str = self._format_time(max(0, game['black_time']))
                timer_text = f"{white_time_str} | {black_time_str}"
                
                timer_label = self.timer_labels[game_id]
                timer_label.config(text=timer_text, fg=self._get_timer_color(game_id))
                
                # Update accuracy display
                white_acc = game['white_accuracy']
                black_acc = game['black_accuracy']
                acc_text = f"W: {white_acc:.0f}% | B: {black_acc:.0f}%"
                self.accuracy_labels[game_id].config(text=acc_text)
                
                # Update result if game finished
                if game['result']:
                    result_color = "green" if game['result'] == "Win" else ("red" if game['result'] == "Loss" else "orange")
                    self.result_labels[game_id].config(text=game['result'], fg=result_color)
                else:
                    self.result_labels[game_id].config(text="", fg="black")
        
        # Schedule next refresh
        if self.running:
            self.root.after(100, self.refresh)
    
    def set_status(self, text: str):
        """Update status bar."""
        self.status_label.config(text=text)
    
    def set_epoch(self, epoch: int):
        """Update epoch display."""
        self.epoch_label.config(text=f"Epoch: {epoch}")
    
    def run(self):
        """Start the GUI event loop."""
        self.refresh()
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            self.stop()
    
    def stop(self):
        """Stop the visualizer."""
        self.running = False
        try:
            self.root.quit()
        except:
            pass
    
    def close(self):
        """Properly close the GUI."""
        self.running = False
        try:
            self.root.destroy()
        except:
            pass


def launch_visualizer(num_games: int = 28) -> GameVisualizerGUI:
    """
    Launch the game visualizer in a separate thread.
    
    Args:
        num_games: Number of games to visualize
        
    Returns:
        GameVisualizerGUI instance
    """
    visualizer = GameVisualizerGUI(num_games=num_games)
    
    # Run in separate thread
    gui_thread = Thread(target=visualizer.run, daemon=True)
    gui_thread.start()
    
    return visualizer


if __name__ == "__main__":
    # Test the visualizer
    visualizer = GameVisualizerGUI(num_games=28)
    
    # Simulate some game updates for testing
    import random
    
    def test_updates():
        for i in range(28):
            board = chess.Board()
            white_time = 60.0 - random.random() * 20
            black_time = 60.0 - random.random() * 20
            moves = random.randint(0, 50)
            white_acc = random.random() * 100
            black_acc = random.random() * 100
            
            visualizer.update_game(
                i, board, white_time, black_time, moves,
                white_acc, black_acc, random.uniform(-1, 1)
            )
        
        visualizer.set_status(f"Test Mode - {28} games")
        visualizer.set_epoch(1)
    
    test_updates()
    visualizer.run()
