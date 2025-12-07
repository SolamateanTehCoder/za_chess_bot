"""
Time management system for tournament chess.
Handles Fischer clock, Bronstein delay, and adaptive time allocation.
Critical for WCCC tournament compliance.
"""

import time
from enum import Enum
from typing import Optional, Tuple
from dataclasses import dataclass


class ClockType(Enum):
    """Time control types."""
    FISCHER = "fischer"  # Increment happens after move
    BRONSTEIN = "bronstein"  # Increment happens before move
    SUDDEN_DEATH = "sudden_death"  # No increment


@dataclass
class TimeControl:
    """Time control specification."""
    initial_time_ms: int  # Initial time in milliseconds
    increment_ms: int  # Time increment per move
    clock_type: ClockType = ClockType.FISCHER
    
    def __init__(self, initial_minutes: float = 90, increment_seconds: float = 0,
                 clock_type: ClockType = ClockType.FISCHER):
        """
        Initialize time control.
        
        Args:
            initial_minutes: Initial time in minutes
            increment_seconds: Increment per move in seconds
            clock_type: Type of clock
        """
        self.initial_time_ms = int(initial_minutes * 60 * 1000)
        self.increment_ms = int(increment_seconds * 1000)
        self.clock_type = clock_type


class ChessClock:
    """Chess clock manager for tournament play."""
    
    def __init__(self, time_control: TimeControl):
        """
        Initialize chess clock.
        
        Args:
            time_control: Time control specification
        """
        self.time_control = time_control
        self.white_time_ms = time_control.initial_time_ms
        self.black_time_ms = time_control.initial_time_ms
        self.white_move_start = None
        self.black_move_start = None
        self.move_count = 0
        self.total_moves = 0
    
    def start_move(self, is_white: bool):
        """
        Start timer for player's move.
        
        Args:
            is_white: True if white to move
        """
        if is_white:
            self.white_move_start = time.time()
        else:
            self.black_move_start = time.time()
    
    def end_move(self, is_white: bool) -> float:
        """
        End move and update clock.
        
        Args:
            is_white: True if white just moved
            
        Returns:
            Time used for move in milliseconds
        """
        if is_white:
            if self.white_move_start is None:
                return 0
            
            move_time = (time.time() - self.white_move_start) * 1000
            
            # Update white's time
            self.white_time_ms -= int(move_time)
            
            # Add increment (Fischer clock)
            if self.time_control.clock_type == ClockType.FISCHER:
                self.white_time_ms += self.time_control.increment_ms
            
            self.white_move_start = None
            self.move_count += 1
            
            return move_time
        else:
            if self.black_move_start is None:
                return 0
            
            move_time = (time.time() - self.black_move_start) * 1000
            
            # Update black's time
            self.black_time_ms -= int(move_time)
            
            # Add increment (Fischer clock)
            if self.time_control.clock_type == ClockType.FISCHER:
                self.black_time_ms += self.time_control.increment_ms
            
            self.black_move_start = None
            self.total_moves += 1
            
            return move_time
    
    def get_time_ms(self, is_white: bool) -> int:
        """
        Get remaining time for player.
        
        Args:
            is_white: True for white
            
        Returns:
            Remaining time in milliseconds
        """
        return self.white_time_ms if is_white else self.black_time_ms
    
    def is_time_up(self, is_white: bool) -> bool:
        """
        Check if player has run out of time.
        
        Args:
            is_white: True for white
            
        Returns:
            True if time has expired
        """
        remaining = self.get_time_ms(is_white)
        return remaining <= 0
    
    def __str__(self) -> str:
        """String representation of clock state."""
        w_min = self.white_time_ms // 60000
        w_sec = (self.white_time_ms % 60000) // 1000
        b_min = self.black_time_ms // 60000
        b_sec = (self.black_time_ms % 60000) // 1000
        
        return f"White: {w_min}:{w_sec:02d}  Black: {b_min}:{b_sec:02d}"


class TimeAllocator:
    """
    Allocate time for moves based on remaining time and position complexity.
    Uses algorithms similar to Stockfish and modern engines.
    """
    
    def __init__(self, total_time_ms: int, increment_ms: int = 0):
        """
        Initialize time allocator.
        
        Args:
            total_time_ms: Total time available
            increment_ms: Time increment per move
        """
        self.total_time_ms = total_time_ms
        self.increment_ms = increment_ms
        self.moves_made = 0
    
    def allocate_time(self, remaining_time_ms: int, moves_left: Optional[int] = None,
                     position_complexity: float = 0.5) -> int:
        """
        Allocate time for next move.
        
        Args:
            remaining_time_ms: Remaining time for this side
            moves_left: Estimated moves to end of time control (None = unknown)
            position_complexity: Position complexity 0.0 (simple) to 1.0 (complex)
            
        Returns:
            Time allocation for next move in milliseconds
        """
        # Ensure minimum time
        if remaining_time_ms < 1000:
            return max(100, remaining_time_ms // 2)
        
        # Estimate moves to endgame time control
        if moves_left is None:
            # Conservative estimate: assume 40 more moves
            moves_left = max(5, 40 - self.moves_made)
        
        # Base allocation
        base_allocation = remaining_time_ms / (moves_left + 2)
        
        # Adjust for increment
        if self.increment_ms > 0:
            # With increment, can afford to use more time
            base_allocation += self.increment_ms * 0.8
        
        # Adjust for position complexity
        # Simple positions: quicker moves (complexity close to 0)
        # Complex positions: more thinking (complexity close to 1)
        complexity_factor = 0.5 + (position_complexity * 0.5)  # Range [0.5, 1.0]
        adjusted_allocation = base_allocation * complexity_factor
        
        # Safeguard against using too much time at once
        max_allocation = remaining_time_ms * 0.25  # Never use more than 25% at once
        
        allocation = min(adjusted_allocation, max_allocation)
        
        # Minimum allocation
        allocation = max(100, int(allocation))
        
        self.moves_made += 1
        
        return allocation
    
    def allocate_time_opening(self, remaining_time_ms: int, move_number: int) -> int:
        """
        Allocate time for opening moves (faster decisions expected).
        
        Args:
            remaining_time_ms: Remaining time
            move_number: Current move number (1-indexed)
            
        Returns:
            Time allocation in milliseconds
        """
        # Opening moves should be quick (prepared lines)
        opening_threshold = 15  # First 15 moves
        
        if move_number <= opening_threshold:
            # Quick allocation for opening
            allocation = remaining_time_ms / (40 - move_number + 2)
            allocation += self.increment_ms * 0.5 if self.increment_ms > 0 else 0
            return max(100, int(allocation))
        else:
            # Transition to middlegame allocation
            return self.allocate_time(remaining_time_ms, moves_left=30)
    
    def allocate_time_endgame(self, remaining_time_ms: int, material_count: int,
                             is_winning: bool) -> int:
        """
        Allocate time for endgame moves.
        
        Args:
            remaining_time_ms: Remaining time
            material_count: Total pieces remaining (for complexity)
            is_winning: Whether position is winning for us
            
        Returns:
            Time allocation in milliseconds
        """
        # Endgames need more time due to precision
        endgame_threshold = 7  # 7 or fewer pieces
        
        # Estimate moves to mate or draw
        if is_winning:
            # Winning endgame: need to find forcing moves
            estimated_moves = max(10, 30 - material_count * 3)
        else:
            # Defensive endgame: careful play needed
            estimated_moves = max(15, 50 - material_count * 2)
        
        allocation = remaining_time_ms / (estimated_moves + 1)
        allocation += self.increment_ms if self.increment_ms > 0 else 0
        
        return max(200, int(allocation))


class TimeManager:
    """High-level time manager combining clock and allocation."""
    
    def __init__(self, time_control: TimeControl):
        """
        Initialize time manager.
        
        Args:
            time_control: Time control specification
        """
        self.clock = ChessClock(time_control)
        self.allocator = TimeAllocator(
            time_control.initial_time_ms,
            time_control.increment_ms
        )
    
    def get_move_time_allocation(self, is_white: bool, move_number: int,
                                position_complexity: float = 0.5) -> int:
        """
        Get recommended time for next move.
        
        Args:
            is_white: True if white to move
            move_number: Current move number (1-indexed)
            position_complexity: Position complexity (0.0-1.0)
            
        Returns:
            Recommended time in milliseconds
        """
        remaining_time = self.clock.get_time_ms(is_white)
        
        # Adjust for position phase
        if move_number <= 15:
            return self.allocator.allocate_time_opening(remaining_time, move_number)
        elif move_number >= 40:
            # Estimate material (placeholder)
            material = 32 - (move_number // 5)  # Rough estimate
            return self.allocator.allocate_time_endgame(remaining_time, material, False)
        else:
            return self.allocator.allocate_time(remaining_time,
                                               position_complexity=position_complexity)
