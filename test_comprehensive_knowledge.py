"""Test the comprehensive chess knowledge system."""
import chess
from comprehensive_chess_knowledge import ComprehensiveChessKnowledge

def test_opening_book():
    """Test the opening book with various positions."""
    knowledge = ComprehensiveChessKnowledge()
    board = chess.Board()
    
    print("=" * 60)
    print("TESTING OPENING BOOK")
    print("=" * 60)
    
    # Test first few moves
    for move_num in range(1, 11):
        move, source = knowledge.get_assisted_move(board)
        if move:
            print(f"Move {move_num}: {move.uci()} (from {source})")
            board.push(move)
        else:
            print(f"Move {move_num}: No opening book move found")
            break
    
    print(f"\nFinal position after {board.fullmove_number} moves:")
    print(board)
    print(f"\nStatistics:")
    print(f"  Opening book moves used: {knowledge.opening_book.stats['moves_used']}")

def test_multiple_openings():
    """Test different opening variations."""
    knowledge = ComprehensiveChessKnowledge()
    
    print("\n" + "=" * 60)
    print("TESTING MULTIPLE OPENINGS")
    print("=" * 60)
    
    # Test a few different opening sequences
    test_sequences = [
        ("e4 e5", "King's Pawn Opening"),
        ("d4 d5", "Queen's Pawn Opening"),
        ("Nf3 Nf6", "Reti Opening"),
        ("c4 e5", "English Opening"),
    ]
    
    for sequence, name in test_sequences:
        board = chess.Board()
        print(f"\n{name}: {sequence}")
        
        for move_san in sequence.split():
            try:
                board.push_san(move_san)
            except:
                pass
        
        move, source = knowledge.get_assisted_move(board)
        if move:
            print(f"  Suggested continuation: {move.uci()} (from {source})")
        else:
            print(f"  No suggestion found")

def test_tactical_awareness():
    """Test tactical pattern recognition."""
    knowledge = ComprehensiveChessKnowledge()
    
    print("\n" + "=" * 60)
    print("TESTING TACTICAL AWARENESS")
    print("=" * 60)
    
    # Create a position with tactical elements
    board = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1")
    print("\nPosition (Italian Game with tactical elements):")
    print(board)
    
    move, source = knowledge.get_assisted_move(board)
    print(f"\nSuggestion: {move.uci() if move else 'None'} (from {source})")
    
    tactics = knowledge.tactics
    print(f"\nAvailable tactical patterns: {len(tactics.TACTICS)}")
    print("Sample tactics:")
    for tactic_name in list(tactics.TACTICS.keys())[:5]:
        print(f"  - {tactic_name}: {tactics.TACTICS[tactic_name][:60]}...")

def test_endgame_knowledge():
    """Test endgame recognition and advice."""
    knowledge = ComprehensiveChessKnowledge()
    
    print("\n" + "=" * 60)
    print("TESTING ENDGAME KNOWLEDGE")
    print("=" * 60)
    
    # Create a simple endgame position
    board = chess.Board("8/8/4k3/8/8/4K3/8/7R w - - 0 1")
    print("\nPosition (King and Rook vs King):")
    print(board)
    
    is_endgame = knowledge.endgame.is_endgame(board)
    print(f"\nIs endgame: {is_endgame}")
    
    if is_endgame:
        plan = knowledge.endgame.suggest_plan(board)
        print(f"Suggested plan: {plan}")
    
    move, source = knowledge.get_assisted_move(board)
    print(f"Suggested move: {move.uci() if move else 'None'} (from {source})")
    
    print(f"\nEndgame principles available: {len(knowledge.endgame.ENDGAME_PRINCIPLES)}")

def test_statistics():
    """Test the statistics tracking."""
    knowledge = ComprehensiveChessKnowledge()
    
    print("\n" + "=" * 60)
    print("TESTING STATISTICS TRACKING")
    print("=" * 60)
    
    # Play through an opening
    board = chess.Board()
    for _ in range(10):
        move, source = knowledge.get_assisted_move(board)
        if move:
            board.push(move)
        else:
            break
    
    stats = knowledge.get_stats()
    print("\nKnowledge system statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

def main():
    """Run all tests."""
    test_opening_book()
    test_multiple_openings()
    test_tactical_awareness()
    test_endgame_knowledge()
    test_statistics()
    
    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED")
    print("=" * 60)

if __name__ == "__main__":
    main()
