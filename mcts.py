"""
Monte Carlo Tree Search (MCTS) for chess move selection.
"""

import math
import random
import chess
from typing import Optional
import torch

class MCTSNode:
    """A node in the Monte Carlo Tree Search."""
    def __init__(self, board: chess.Board, parent: Optional['MCTSNode'] = None, move: Optional[chess.Move] = None, policy: float = 0.0):
        self.board = board
        self.parent = parent
        self.move = move
        self.children = []
        self.wins = 0
        self.visits = 0
        self.unexplored_moves = list(board.legal_moves)
        self.policy = policy

    def select_child(self, exploration_param: float = 1.414) -> 'MCTSNode':
        """Select a child node using the UCB1 formula."""
        best_child = max(self.children, key=lambda c: c.wins / c.visits + exploration_param * self.policy * math.sqrt(math.log(self.visits) / (1 + c.visits)))
        return best_child

    def expand(self, policy_logits: torch.Tensor) -> 'MCTSNode':
        """Expand the tree by creating a new child node."""
        move = self.unexplored_moves.pop()

        # Get policy for this move
        policy_index = move.from_square * 64 + move.to_square
        policy = policy_logits[0, policy_index].item()

        new_board = self.board.copy()
        new_board.push(move)
        child_node = MCTSNode(new_board, parent=self, move=move, policy=policy)
        self.children.append(child_node)
        return child_node

    def update(self, result: float):
        """Update the node's statistics."""
        self.visits += 1
        self.wins += result

    def is_fully_expanded(self) -> bool:
        """Check if the node has been fully expanded."""
        return len(self.unexplored_moves) == 0

class MCTS:
    """Monte Carlo Tree Search algorithm."""
    def __init__(self, player, exploration_param: float = 1.414):
        self.player = player
        self.exploration_param = exploration_param

    def search(self, board: chess.Board, num_simulations: int) -> chess.Move:
        """Perform MCTS and return the best move."""
        root = MCTSNode(board)

        for _ in range(num_simulations):
            node = root
            # Selection
            while not node.board.is_game_over() and node.is_fully_expanded():
                node = node.select_child(self.exploration_param)

            # Expansion
            if not node.board.is_game_over():
                policy_logits, value = self.player.model(self.player.encode_board(node.board).unsqueeze(0))
                node = node.expand(policy_logits)

            # Simulation (now using NN evaluation)
            _, value = self.player.model(self.player.encode_board(node.board).unsqueeze(0))
            result = value.item()

            # Backpropagation
            while node is not None:
                node.update(result)
                node = node.parent

        best_child = max(root.children, key=lambda c: c.visits)
        return best_child.move
