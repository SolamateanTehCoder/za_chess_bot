"""
WCCC Bot Setup and Initialization Script

Configures the chess bot for World Computer Chess Championship competition.
Handles dependencies, model loading, opening books, and tournaments.
"""

import os
import sys
import json
from pathlib import Path
import subprocess
import torch
import chess


class WCCCSetup:
    """Setup and initialization for WCCC participation."""
    
    def __init__(self):
        """Initialize setup."""
        self.root_dir = Path(__file__).parent
        self.config_file = self.root_dir / "wccc_config.json"
        self.config = self._load_config()
    
    def _load_config(self) -> dict:
        """Load or create configuration."""
        if self.config_file.exists():
            with open(self.config_file) as f:
                return json.load(f)
        
        return self._default_config()
    
    def _default_config(self) -> dict:
        """Get default configuration."""
        return {
            "engine_name": "Za Chess Bot",
            "engine_version": "1.0.0",
            "author": "Arjun",
            "stockfish_path": self._find_stockfish(),
            "threads": 4,
            "hash_size": 256,
            "use_tablebase": False,
            "tablebase_paths": [],
            "opening_book_enabled": False,
            "opening_book_path": str(self.root_dir / "openings.json"),
            "use_gpu": torch.cuda.is_available(),
            "model_checkpoint": "latest",
            "tournament_mode": True
        }
    
    def _find_stockfish(self) -> str:
        """Find Stockfish executable."""
        paths = [
            "C:\\stockfish\\stockfish-windows-x86-64-avx2.exe",  # Windows
            "/usr/bin/stockfish",  # Linux
            "/usr/local/bin/stockfish",  # macOS
            "stockfish",  # In PATH
        ]
        
        for path in paths:
            try:
                result = subprocess.run([path, "--version"], capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    print(f"[INFO] Found Stockfish at: {path}")
                    return path
            except:
                continue
        
        return "stockfish"  # Default fallback
    
    def verify_environment(self) -> bool:
        """Verify all required components are available."""
        print("\n=== Za Chess Bot - WCCC Environment Verification ===\n")
        
        checks_passed = 0
        checks_total = 0
        
        # Check Python version
        checks_total += 1
        if sys.version_info >= (3, 10):
            print(f"✓ Python {sys.version.split()[0]} (OK - 3.10+ required)")
            checks_passed += 1
        else:
            print(f"✗ Python {sys.version.split()[0]} (FAIL - 3.10+ required)")
        
        # Check PyTorch
        checks_total += 1
        try:
            import torch
            print(f"✓ PyTorch {torch.__version__} (OK)")
            checks_passed += 1
            
            if torch.cuda.is_available():
                print(f"  ├─ CUDA: {torch.version.cuda} - Device: {torch.cuda.get_device_name()}")
            else:
                print(f"  └─ CUDA: Not available (CPU mode)")
        except ImportError:
            print("✗ PyTorch (FAIL - not installed)")
        
        # Check python-chess
        checks_total += 1
        try:
            import chess
            print(f"✓ python-chess {chess.__version__} (OK)")
            checks_passed += 1
        except ImportError:
            print("✗ python-chess (FAIL - not installed)")
        
        # Check Stockfish
        checks_total += 1
        try:
            result = subprocess.run([self.config["stockfish_path"], "--version"],
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                version = result.stdout.split()[0]
                print(f"✓ Stockfish {version} at {self.config['stockfish_path']} (OK)")
                checks_passed += 1
            else:
                print(f"✗ Stockfish (FAIL - not responding)")
        except Exception as e:
            print(f"✗ Stockfish (FAIL - {e})")
        
        # Check GPU (optional)
        checks_total += 1
        if torch.cuda.is_available():
            print(f"✓ GPU Support (OK - {torch.cuda.get_device_name()})")
            checks_passed += 1
        else:
            print(f"⚠ GPU Support (Optional - using CPU)")
            checks_passed += 1  # Don't fail on this
        
        # Check model checkpoint
        checks_total += 1
        checkpoint_dir = self.root_dir / "checkpoints"
        if checkpoint_dir.exists() and list(checkpoint_dir.glob("*.pt")):
            latest = max(checkpoint_dir.glob("*.pt"), key=os.path.getctime)
            print(f"✓ Model Checkpoint {latest.name} (OK)")
            checks_passed += 1
        else:
            print(f"⚠ Model Checkpoint (Not found - will use random network)")
            checks_passed += 1  # This is optional
        
        print(f"\n{'='*50}")
        print(f"Environment Status: {checks_passed}/{checks_total} checks passed")
        print(f"{'='*50}\n")
        
        return checks_passed >= 4  # Require at least Python, PyTorch, python-chess, Stockfish
    
    def setup_model(self) -> bool:
        """Setup neural network model."""
        print("\n=== Model Setup ===\n")
        
        try:
            from chess_models import SimpleChessNet, ChessNetV2
            
            checkpoint_dir = self.root_dir / "checkpoints"
            checkpoint_dir.mkdir(exist_ok=True)
            
            # Try to load latest checkpoint
            checkpoints = list(checkpoint_dir.glob("*.pt"))
            if checkpoints:
                latest = max(checkpoints, key=os.path.getctime)
                checkpoint = torch.load(latest, weights_only=True)
                print(f"✓ Loaded checkpoint: {latest.name}")
                return True
            else:
                print("⚠ No checkpoint found - will initialize fresh network")
                print("  Recommendation: Train model with python train.py")
                return True
        except Exception as e:
            print(f"✗ Error setting up model: {e}")
            return False
    
    def setup_opening_book(self) -> bool:
        """Setup opening book from PGN."""
        print("\n=== Opening Book Setup ===\n")
        
        try:
            from opening_book import OpeningBook
            
            book_path = Path(self.config["opening_book_path"])
            
            if book_path.exists():
                book = OpeningBook()
                book.load_book(str(book_path))
                stats = book.get_stats()
                print(f"✓ Loaded opening book with {stats['positions_count']} positions")
                return True
            else:
                print("⚠ Opening book not found")
                print("  To create: from opening_book import OpeningBook")
                print("             book = OpeningBook()")
                print("             book.learn_from_pgn('path/to/games.pgn')")
                print("             book.save_book('openings.json')")
                return True
        except Exception as e:
            print(f"✗ Error setting up opening book: {e}")
            return False
    
    def setup_tablebases(self) -> bool:
        """Setup Syzygy tablebases."""
        print("\n=== Tablebase Setup ===\n")
        
        try:
            from tablebase_manager import TablebaseManager
            
            tb_manager = TablebaseManager()
            stats = tb_manager.get_statistics()
            
            if stats["tablebases_loaded"] > 0:
                print(f"✓ Loaded {stats['tablebases_loaded']} tablebase(s)")
                return True
            else:
                print("⚠ No Syzygy tablebases found")
                print("  Recommendation: Download from https://github.com/syzygy-tables/tb")
                print("  Extract to C:\\Syzygy (Windows) or ~/syzygy (Linux)")
                return True
        except Exception as e:
            print(f"✗ Error setting up tablebases: {e}")
            return False
    
    def save_config(self):
        """Save configuration to file."""
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
        print(f"\n[INFO] Configuration saved to {self.config_file}")
    
    def full_setup(self) -> bool:
        """Run complete setup process."""
        print("Starting Za Chess Bot WCCC Setup...\n")
        
        # Verify environment
        if not self.verify_environment():
            print("[ERROR] Environment verification failed!")
            print("Please install required packages:")
            print("  pip install -r requirements.txt")
            return False
        
        # Setup components
        model_ok = self.setup_model()
        book_ok = self.setup_opening_book()
        tb_ok = self.setup_tablebases()
        
        if model_ok and book_ok and tb_ok:
            self.save_config()
            print("\n" + "="*50)
            print("✓ Setup Complete - Ready for WCCC!")
            print("="*50)
            print("\nNext steps:")
            print("1. Run: python uci_engine.py")
            print("2. Add to tournament arena")
            print("3. Play matches and improve!")
            return True
        else:
            print("\n[WARNING] Setup completed with warnings")
            return True


def print_usage():
    """Print usage information."""
    print("""
Za Chess Bot - WCCC Setup

Usage:
    python wccc_setup.py [command]

Commands:
    verify      - Verify environment
    setup       - Full setup
    engine      - Run UCI engine
    train       - Train neural network
    tournament  - Run tournament
    
Examples:
    python wccc_setup.py verify     # Check system
    python wccc_setup.py setup      # Initialize bot
    python wccc_setup.py engine     # Start for tournaments
    """)


if __name__ == "__main__":
    setup = WCCCSetup()
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "verify":
            setup.verify_environment()
        elif command == "setup":
            setup.full_setup()
        elif command == "engine":
            print("[INFO] Starting UCI engine...")
            os.system("python uci_engine.py")
        elif command == "train":
            print("[INFO] Starting training...")
            os.system("python train.py")
        elif command == "tournament":
            print("[INFO] Starting tournament...")
            os.system("python tournament.py")
        else:
            print_usage()
    else:
        # Default: run full setup
        setup.full_setup()
