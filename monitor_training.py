"""
Distributed Training Monitor - Real-time dashboard for game generation and training
"""
import json
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import urllib.request
from collections import defaultdict


class GitHubMonitor:
    """Monitor GitHub Actions and releases"""
    
    def __init__(self, owner: str, repo: str):
        self.owner = owner
        self.repo = repo
        self.base_url = f"https://api.github.com/repos/{owner}/{repo}"
    
    def get_latest_releases(self, limit: int = 5) -> List[Dict]:
        """Get latest game releases"""
        try:
            url = f"{self.base_url}/releases"
            with urllib.request.urlopen(url) as response:
                releases = json.loads(response.read().decode())
                game_releases = [r for r in releases if r['tag_name'].startswith('games-')]
                return game_releases[:limit]
        except Exception as e:
            print(f"Error fetching releases: {e}")
            return []
    
    def get_workflow_runs(self, workflow_id: str = "game_generation.yml", limit: int = 10) -> List[Dict]:
        """Get recent workflow runs"""
        try:
            url = f"{self.base_url}/actions/workflows/{workflow_id}/runs?per_page={limit}"
            with urllib.request.urlopen(url) as response:
                data = json.loads(response.read().decode())
                return data.get('workflow_runs', [])
        except Exception as e:
            print(f"Error fetching workflow runs: {e}")
            return []
    
    def get_repo_info(self) -> Optional[tuple]:
        """Get repo owner/repo from git remote"""
        try:
            remote_url = subprocess.check_output(
                ["git", "config", "--get", "remote.origin.url"],
                text=True
            ).strip()
            
            if "github.com" in remote_url:
                if remote_url.startswith("git@"):
                    parts = remote_url.split(":")[-1].replace(".git", "").split("/")
                else:
                    parts = remote_url.rstrip("/").replace(".git", "").split("/")
                
                return (parts[-2], parts[-1])
        except Exception as e:
            print(f"Error getting repo info: {e}")
        
        return None


class LocalMonitor:
    """Monitor local game and training progress"""
    
    def __init__(self):
        self.batch_dir = Path("game_batches")
        self.merged_file = Path("merged_games.jsonl")
        self.checkpoint_dir = Path("checkpoints")
    
    def get_batch_stats(self) -> Dict:
        """Get statistics for downloaded game batches"""
        stats = {
            'batch_count': 0,
            'total_games': 0,
            'total_size_mb': 0.0,
            'latest_batch': None,
            'batches': []
        }
        
        if not self.batch_dir.exists():
            return stats
        
        batch_files = list(self.batch_dir.glob("games_batch_*.jsonl"))
        stats['batch_count'] = len(batch_files)
        
        for batch_file in sorted(batch_files, reverse=True):
            try:
                size_mb = batch_file.stat().st_size / (1024 * 1024)
                with open(batch_file, 'r', encoding='utf-8') as f:
                    game_count = sum(1 for _ in f if _.strip())
                
                stats['total_games'] += game_count
                stats['total_size_mb'] += size_mb
                
                if not stats['latest_batch']:
                    stats['latest_batch'] = {
                        'name': batch_file.name,
                        'size_mb': size_mb,
                        'games': game_count
                    }
                
                stats['batches'].append({
                    'name': batch_file.name,
                    'size_mb': size_mb,
                    'games': game_count,
                    'modified': datetime.fromtimestamp(batch_file.stat().st_mtime).isoformat()
                })
            except Exception as e:
                print(f"Error reading {batch_file}: {e}")
        
        return stats
    
    def get_merged_games_stats(self) -> Dict:
        """Get statistics for merged games file"""
        stats = {
            'exists': self.merged_file.exists(),
            'size_mb': 0.0,
            'game_count': 0,
            'modified': None
        }
        
        if stats['exists']:
            try:
                stats['size_mb'] = self.merged_file.stat().st_size / (1024 * 1024)
                stats['modified'] = datetime.fromtimestamp(
                    self.merged_file.stat().st_mtime
                ).isoformat()
                
                with open(self.merged_file, 'r', encoding='utf-8') as f:
                    stats['game_count'] = sum(1 for _ in f if _.strip())
            except Exception as e:
                print(f"Error reading merged games: {e}")
        
        return stats
    
    def get_checkpoint_stats(self) -> Dict:
        """Get statistics for model checkpoints"""
        stats = {
            'checkpoint_count': 0,
            'latest_checkpoint': None,
            'total_size_mb': 0.0,
            'checkpoints': []
        }
        
        if not self.checkpoint_dir.exists():
            return stats
        
        checkpoints = list(self.checkpoint_dir.glob("model_checkpoint_*.pt"))
        stats['checkpoint_count'] = len(checkpoints)
        
        for checkpoint in sorted(checkpoints, reverse=True):
            try:
                size_mb = checkpoint.stat().st_size / (1024 * 1024)
                stats['total_size_mb'] += size_mb
                
                checkpoint_info = {
                    'name': checkpoint.name,
                    'size_mb': size_mb,
                    'modified': datetime.fromtimestamp(checkpoint.stat().st_mtime).isoformat()
                }
                
                # Extract game number if available
                stem = checkpoint.stem
                if 'game_' in stem:
                    try:
                        game_num = int(stem.split('game_')[-1])
                        checkpoint_info['game_number'] = game_num
                    except:
                        pass
                
                if not stats['latest_checkpoint']:
                    stats['latest_checkpoint'] = checkpoint_info
                
                stats['checkpoints'].append(checkpoint_info)
            except Exception as e:
                print(f"Error reading {checkpoint}: {e}")
        
        return stats


class Dashboard:
    """Interactive dashboard for monitoring distributed training"""
    
    def __init__(self):
        self.github_monitor = None
        self.local_monitor = LocalMonitor()
        self.refresh_interval = 30  # seconds
        
        # Initialize GitHub monitor if in a repo
        try:
            repo_info = GitHubMonitor(None, None).get_repo_info()
            if repo_info:
                self.github_monitor = GitHubMonitor(repo_info[0], repo_info[1])
        except:
            pass
    
    def print_header(self):
        """Print dashboard header"""
        print("\033[2J\033[H")  # Clear screen and move to top
        print("=" * 80)
        print("  DISTRIBUTED CHESS BOT TRAINING - MONITORING DASHBOARD".center(80))
        print("=" * 80)
        print(f"  Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".ljust(80))
        print("=" * 80)
    
    def print_github_status(self):
        """Print GitHub status if available"""
        if not self.github_monitor:
            print("\n[GitHub] Not connected to repository\n")
            return
        
        print("\n[GitHub Actions Game Generation]\n")
        
        runs = self.github_monitor.get_workflow_runs(limit=5)
        if runs:
            print(f"  Recent Runs (Last 5):")
            for i, run in enumerate(runs[:5], 1):
                status = run['status']
                conclusion = run.get('conclusion', 'pending')
                
                # Status indicator
                if conclusion == 'success':
                    indicator = "✓"
                elif conclusion == 'failure':
                    indicator = "✗"
                else:
                    indicator = "⊙"
                
                created = datetime.fromisoformat(run['created_at'].replace('Z', '+00:00'))
                time_ago = datetime.now(created.tzinfo) - created
                
                print(f"    {i}. {indicator} {run['name']}")
                print(f"       Status: {status} | Result: {conclusion}")
                print(f"       Started: {time_ago.total_seconds() / 3600:.1f} hours ago")
        
        releases = self.github_monitor.get_latest_releases(limit=3)
        if releases:
            print(f"\n  Latest Game Releases (Last 3):")
            for i, release in enumerate(releases[:3], 1):
                created = datetime.fromisoformat(release['created_at'].replace('Z', '+00:00'))
                time_ago = datetime.now(created.tzinfo) - created
                
                asset_count = len(release.get('assets', []))
                print(f"    {i}. {release['tag_name']}")
                print(f"       Assets: {asset_count} | Created: {time_ago.total_seconds() / 3600:.1f}h ago")
    
    def print_local_status(self):
        """Print local training status"""
        print("\n[Local Training Progress]\n")
        
        # Game batches
        batch_stats = self.get_batch_stats()
        print(f"  Downloaded Game Batches:")
        print(f"    Count: {batch_stats['batch_count']}")
        print(f"    Total Games: {batch_stats['total_games']:,}")
        print(f"    Total Size: {batch_stats['total_size_mb']:.1f}MB")
        if batch_stats['latest_batch']:
            print(f"    Latest: {batch_stats['latest_batch']['name']} "
                  f"({batch_stats['latest_batch']['size_mb']:.1f}MB, "
                  f"{batch_stats['latest_batch']['games']} games)")
        
        # Merged games
        merged_stats = self.local_monitor.get_merged_games_stats()
        print(f"\n  Merged Training Data:")
        if merged_stats['exists']:
            print(f"    Games: {merged_stats['game_count']:,}")
            print(f"    Size: {merged_stats['size_mb']:.1f}MB")
            print(f"    Last Updated: {merged_stats['modified']}")
        else:
            print(f"    Not created yet (will be created on first training run)")
        
        # Checkpoints
        checkpoint_stats = self.local_monitor.get_checkpoint_stats()
        print(f"\n  Model Checkpoints:")
        print(f"    Count: {checkpoint_stats['checkpoint_count']}")
        print(f"    Total Size: {checkpoint_stats['total_size_mb']:.1f}MB")
        if checkpoint_stats['latest_checkpoint']:
            latest = checkpoint_stats['latest_checkpoint']
            print(f"    Latest: {latest['name']} ({latest['size_mb']:.1f}MB)")
            if 'game_number' in latest:
                print(f"            Game #{latest['game_number']:,}")
            print(f"            Modified: {latest['modified']}")
    
    def print_stats(self):
        """Print combined statistics"""
        print("\n[System Status]\n")
        
        batch_stats = self.local_monitor.get_batch_stats()
        merged_stats = self.local_monitor.get_merged_games_stats()
        checkpoint_stats = self.local_monitor.get_checkpoint_stats()
        
        # Calculate rates and projections
        total_games = batch_stats['total_games'] + merged_stats['game_count']
        total_size = batch_stats['total_size_mb'] + merged_stats['size_mb']
        
        print(f"  Total Games Available: {total_games:,}")
        print(f"  Total Disk Space Used: {total_size:.1f}MB")
        print(f"  Average Game Size: {(total_size / max(total_games, 1)):.2f}MB")
        
        # Projection for continuous operation
        if batch_stats['batch_count'] > 0:
            games_per_batch = batch_stats['total_games'] / batch_stats['batch_count']
            print(f"  Average Games per Batch: {games_per_batch:.0f}")
            print(f"  (At 5-hour intervals, ~{games_per_batch * (24/5):.0f} games/day)")
    
    def print_recommendations(self):
        """Print recommendations based on current status"""
        print("\n[Recommendations]\n")
        
        batch_stats = self.local_monitor.get_batch_stats()
        merged_stats = self.local_monitor.get_merged_games_stats()
        
        if batch_stats['total_games'] == 0:
            print("  ⚠ No game batches downloaded yet")
            print("    → Run: python train_with_batches.py")
            print("    → Or trigger GitHub Actions manually")
        elif merged_stats['game_count'] == 0:
            print("  ✓ Game batches available for training")
            print("    → Run: python train_with_batches.py")
        else:
            print("  ✓ Training data is up to date")
            games_ready = batch_stats['total_games'] - merged_stats['game_count']
            if games_ready > 100:
                print(f"  ⚠ {games_ready:,} unmerged games available")
                print("    → Consider running training coordinator again")
        
        checkpoint_stats = self.local_monitor.get_checkpoint_stats()
        if checkpoint_stats['checkpoint_count'] == 0:
            print("  ⚠ No model checkpoints found")
            print("    → Running training will create the first checkpoint")
    
    def run(self, interval: int = 30, continuous: bool = True):
        """Run dashboard"""
        try:
            while True:
                self.print_header()
                self.print_github_status()
                self.print_local_status()
                self.print_stats()
                self.print_recommendations()
                
                print("\n" + "=" * 80)
                if continuous:
                    print(f"  Press Ctrl+C to exit. Refreshing in {interval}s...".center(80))
                    print("=" * 80)
                    time.sleep(interval)
                else:
                    print("=" * 80)
                    break
        except KeyboardInterrupt:
            print("\n\nDashboard closed.")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Monitor distributed chess bot training"
    )
    parser.add_argument(
        "--interval", "-i",
        type=int,
        default=30,
        help="Refresh interval in seconds (default: 30)"
    )
    parser.add_argument(
        "--once", "-o",
        action="store_true",
        help="Show status once and exit"
    )
    
    args = parser.parse_args()
    
    dashboard = Dashboard()
    dashboard.run(interval=args.interval, continuous=not args.once)


if __name__ == "__main__":
    main()
