"""
Training Coordinator for GitHub Actions distributed game generation
Pulls game batches from releases, trains on accumulated games, syncs checkpoints back
"""
import json
import subprocess
import sys
import time
import urllib.request
import tempfile
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Dict
import torch
from urllib.parse import urlparse

# Import training infrastructure
from train import load_training_state, SimpleChessNet


class BatchDownloader:
    """Download game batches from GitHub releases"""
    
    @staticmethod
    def get_latest_releases(owner: str, repo: str, limit: int = 10) -> List[Dict]:
        """Get latest game releases from GitHub"""
        url = f"https://api.github.com/repos/{owner}/{repo}/releases"
        try:
            with urllib.request.urlopen(url) as response:
                releases = json.loads(response.read().decode())
                # Filter to game batches only
                game_releases = [r for r in releases if r['tag_name'].startswith('games-')]
                return game_releases[:limit]
        except Exception as e:
            print(f"Error fetching releases: {e}")
            return []
    
    @staticmethod
    def download_batch(asset_url: str, output_path: Path) -> bool:
        """Download a single game batch file"""
        try:
            print(f"  Downloading: {asset_url}")
            urllib.request.urlretrieve(asset_url, output_path)
            file_size_mb = output_path.stat().st_size / (1024 * 1024)
            print(f"  Downloaded: {file_size_mb:.1f}MB")
            return True
        except Exception as e:
            print(f"  Download failed: {e}")
            return False
    
    @staticmethod
    def download_latest_batches(owner: str, repo: str, download_dir: Path) -> List[Path]:
        """Download all unprocessed game batches"""
        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Checking for new game batches...")
        
        download_dir.mkdir(exist_ok=True)
        downloaded_files = []
        
        releases = BatchDownloader.get_latest_releases(owner, repo, limit=5)
        if not releases:
            print("  No game releases found")
            return downloaded_files
        
        print(f"  Found {len(releases)} recent releases")
        
        for release in releases:
            if not release.get('assets'):
                continue
            
            for asset in release['assets']:
                if not asset['name'].endswith('.jsonl'):
                    continue
                
                # Skip if already downloaded
                local_path = download_dir / asset['name']
                if local_path.exists():
                    print(f"  Skipping (already exists): {asset['name']}")
                    continue
                
                # Download the batch
                if BatchDownloader.download_batch(asset['browser_download_url'], local_path):
                    downloaded_files.append(local_path)
        
        print(f"  Downloaded {len(downloaded_files)} new batches")
        return downloaded_files


class GameMerger:
    """Merge game batches into training data"""
    
    @staticmethod
    def merge_batches(batch_files: List[Path], output_file: Path) -> int:
        """Merge multiple game batches into single file"""
        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Merging {len(batch_files)} game batches...")
        
        game_count = 0
        with open(output_file, 'w', encoding='utf-8') as outf:
            for batch_file in batch_files:
                print(f"  Merging: {batch_file.name}")
                try:
                    with open(batch_file, 'r', encoding='utf-8') as inf:
                        batch_games = 0
                        for line in inf:
                            if line.strip():
                                outf.write(line)
                                batch_games += 1
                                game_count += 1
                        print(f"    Added {batch_games} games")
                except Exception as e:
                    print(f"    Error reading batch: {e}")
        
        print(f"  Total games merged: {game_count}")
        return game_count


class TrainingCoordinator:
    """Coordinate training on accumulated game batches"""
    
    def __init__(self, github_token: str = None):
        self.github_token = github_token or ""
        self.batch_cache = Path("game_batches")
        self.merged_games = Path("merged_games.jsonl")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Device: {self.device}")
    
    def get_repo_info(self) -> tuple:
        """Get owner/repo from git remote"""
        try:
            remote_url = subprocess.check_output(
                ["git", "config", "--get", "remote.origin.url"],
                text=True
            ).strip()
            
            # Handle both https and ssh formats for GitHub remotes only
            # SSH format: git@github.com:owner/repo.git
            if remote_url.startswith("git@github.com:"):
                parts = remote_url.split(":", 1)[-1].replace(".git", "").split("/")
                if len(parts) >= 2:
                    return (parts[-2], parts[-1])
            
            # HTTPS (and other URL-like) format: https://github.com/owner/repo.git
            parsed = urlparse(remote_url)
            if parsed.hostname == "github.com":
                path_parts = parsed.path.rstrip("/").replace(".git", "").split("/")
                # path is like /owner/repo
                if len(path_parts) >= 3:
                    return (path_parts[-2], path_parts[-1])
        except Exception as e:
            print(f"Error getting repo info: {e}")
        
        return (None, None)
    
    def train_on_batches(self, duration_hours: float = 5):
        """
        Main training loop:
        1. Download new game batches
        2. Merge into training file
        3. Train model
        4. Push checkpoint back
        """
        print("=" * 80)
        print("TRAINING COORDINATOR - Distributed Game Training")
        print("=" * 80)
        
        owner, repo = self.get_repo_info()
        if not owner or not repo:
            print("Error: Could not determine GitHub repository")
            return False
        
        print(f"Repository: {owner}/{repo}")
        
        # Step 1: Download game batches
        downloader = BatchDownloader()
        new_batches = downloader.download_latest_batches(owner, repo, self.batch_cache)
        
        if not new_batches:
            print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] No new game batches to train on")
            return True
        
        # Step 2: Merge batches
        game_count = GameMerger.merge_batches(new_batches, self.merged_games)
        if game_count == 0:
            print("No games to train on")
            return True
        
        # Step 3: Load training state
        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Loading model and training state...")
        model, start_epoch, checkpoint_info = load_training_state(device=self.device)
        
        print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"  Starting epoch: {start_epoch}")
        if checkpoint_info:
            print(f"  Checkpoint info: {checkpoint_info}")
        
        # Step 4: Train on merged games
        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Training on {game_count} games...")
        print(f"  Duration: {duration_hours} hours")
        
        self._train_loop(model, duration_hours)
        
        # Step 5: Save and push checkpoint
        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Saving checkpoint...")
        self._save_and_push_checkpoint(model)
        
        return True
    
    def _train_loop(self, model, duration_hours: float):
        """Run training loop on merged games"""
        # This is simplified - in practice, you'd load the merged_games.jsonl
        # and run the actual training loop from train.py
        
        start_time = time.time()
        print(f"Training started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        try:
            # Load games
            games = []
            with open(self.merged_games, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        games.append(json.loads(line))
            
            print(f"  Loaded {len(games)} games for training")
            
            # Training loop would go here
            # For now, just run for the specified duration
            while True:
                elapsed = (time.time() - start_time) / 3600
                if elapsed >= duration_hours:
                    print(f"Training duration reached ({elapsed:.1f}h)")
                    break
                
                print(f"  Training: {elapsed:.1f}h/{duration_hours}h")
                time.sleep(60)
        
        except Exception as e:
            print(f"Training error: {e}")
    
    def _save_and_push_checkpoint(self, model):
        """Save checkpoint and push back to repository"""
        try:
            # Save checkpoint
            checkpoint_path = Path("checkpoints") / f"model_checkpoint_distributed_{int(time.time())}.pt"
            checkpoint_path.parent.mkdir(exist_ok=True)
            
            torch.save({
                'model_state_dict': model.state_dict(),
                'timestamp': datetime.now().isoformat(),
                'source': 'distributed_training'
            }, checkpoint_path)
            
            print(f"  Saved: {checkpoint_path}")
            
            # Push to git
            try:
                subprocess.run(["git", "add", str(checkpoint_path)], check=True)
                subprocess.run([
                    "git", "commit", "-m",
                    f"Checkpoint from distributed training - {datetime.now().isoformat()}"
                ], check=True)
                subprocess.run(["git", "push"], check=True)
                print(f"  Pushed checkpoint to repository")
            except subprocess.CalledProcessError as e:
                print(f"  Git push warning: {e}")
        
        except Exception as e:
            print(f"Error saving checkpoint: {e}")


def main():
    """Entry point for training coordinator"""
    import os
    
    # Configuration
    duration_hours = float(os.environ.get('TRAINING_DURATION', '5'))
    github_token = os.environ.get('GITHUB_TOKEN', '')
    
    coordinator = TrainingCoordinator(github_token=github_token)
    success = coordinator.train_on_batches(duration_hours=duration_hours)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
