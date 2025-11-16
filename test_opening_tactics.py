"""Test opening-specific middlegame tactical themes."""
import chess
from comprehensive_chess_knowledge import ComprehensiveChessKnowledge

def test_opening_tactics():
    """Test middlegame tactics for various openings."""
    knowledge = ComprehensiveChessKnowledge()
    
    test_positions = [
        # Sicilian Najdorf
        ("Sicilian Najdorf", "rnbqkb1r/1p2pppp/p2p1n2/8/3NP3/2N5/PPP2PPP/R1BQKB1R w KQkq - 0 6"),
        
        # Sicilian Dragon
        ("Sicilian Dragon", "rnbqk2r/pp2ppbp/3p1np1/8/3NP3/2N1B3/PPP2PPP/R2QKB1R w KQkq - 0 7"),
        
        # Ruy Lopez Closed
        ("Ruy Lopez", "r1bqkbnr/1ppp1ppp/p1n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 4"),
        
        # Italian Game
        ("Italian Game", "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"),
        
        # French Defense
        ("French Defense", "rnbqkbnr/ppp2ppp/4p3/3p4/3PP3/8/PPP2PPP/RNBQKBNR w KQkq - 0 3"),
        
        # King's Indian Defense
        ("King's Indian", "rnbqkb1r/pppppp1p/5np1/8/2PP4/6P1/PP2PP1P/RNBQKBNR b KQkq - 0 3"),
        
        # Queen's Gambit Declined
        ("Queen's Gambit Declined", "rnbqkbnr/ppp2ppp/4p3/3p4/2PP4/8/PP2PPPP/RNBQKBNR w KQkq - 0 3"),
        
        # King's Gambit
        ("King's Gambit", "rnbqkbnr/pppp1ppp/8/4p3/4PP2/8/PPPP2PP/RNBQKBNR b KQkq - 0 2"),
    ]
    
    print("=" * 80)
    print("OPENING-SPECIFIC MIDDLEGAME TACTICAL THEMES")
    print("=" * 80)
    
    for opening_name, fen in test_positions:
        board = chess.Board(fen)
        tactics = knowledge.get_opening_tactical_themes(board)
        
        print(f"\n{'=' * 80}")
        print(f"{opening_name.upper()}")
        print(f"{'=' * 80}")
        print(f"\nPosition:")
        print(board)
        print(f"\nMove {board.fullmove_number}, {'White' if board.turn else 'Black'} to move")
        
        print(f"\nTYPICAL MIDDLEGAME TACTICS:")
        for i, tactic in enumerate(tactics.get("typical_tactics", []), 1):
            print(f"   {i}. {tactic}")
        
        print(f"\nKEY SQUARES TO CONTROL:")
        key_squares = tactics.get("key_squares", [])
        if key_squares:
            print(f"   {', '.join(key_squares)}")
        else:
            print("   None specified")
        
        print(f"\nCHARACTERISTIC SACRIFICES:")
        sacrifices = tactics.get("piece_sacrifices", [])
        if sacrifices:
            for sac in sacrifices:
                print(f"   - {sac}")
        else:
            print("   - None typical")
    
    # Test statistics
    print(f"\n\n{'=' * 80}")
    print("STATISTICS")
    print(f"{'=' * 80}")
    stats = knowledge.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print(f"\n{'=' * 80}")
    print("TEST COMPLETE")
    print(f"{'=' * 80}\n")

if __name__ == "__main__":
    test_opening_tactics()
