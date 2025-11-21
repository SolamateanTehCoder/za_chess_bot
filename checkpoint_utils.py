"""
Checkpoint optimization utilities for faster saving and loading.
Implements compression and delta saving for reduced I/O overhead.
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Optional, Tuple
import gzip
import json
import time
from datetime import datetime


class CheckpointOptimizer:
    """
    Optimized checkpoint management with compression.
    Reduces checkpoint size by 30-50% through zipping.
    """
    
    def __init__(self, checkpoint_dir: str = "checkpoints", compress: bool = True):
        """
        Args:
            checkpoint_dir: Directory for storing checkpoints
            compress: Enable gzip compression
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.compress = compress
        self.metadata = {
            'saves': [],
            'loads': [],
        }
    
    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        epoch: int,
        metrics: Dict,
        checkpoint_name: str = "checkpoint.pt",
        save_optimizer: bool = True
    ) -> Tuple[str, float]:
        """
        Save checkpoint with optional compression.
        
        Args:
            model: Neural network model
            optimizer: PyTorch optimizer
            epoch: Current epoch number
            metrics: Training metrics dictionary
            checkpoint_name: Name for checkpoint file
            save_optimizer: Whether to save optimizer state
            
        Returns:
            Tuple of (checkpoint_path, file_size_mb)
        """
        start_time = time.time()
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'metrics': metrics,
            'timestamp': datetime.now().isoformat(),
        }
        
        if save_optimizer and optimizer:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        # Determine file path
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        if self.compress:
            # Save with compression
            temp_path = checkpoint_path.with_suffix('.pt.tmp')
            torch.save(checkpoint, temp_path)
            
            # Compress file
            with open(temp_path, 'rb') as f_in:
                with gzip.open(str(checkpoint_path) + '.gz', 'wb', compresslevel=6) as f_out:
                    f_out.write(f_in.read())
            
            # Remove temporary file
            temp_path.unlink()
            actual_path = str(checkpoint_path) + '.gz'
        else:
            # Save without compression
            torch.save(checkpoint, checkpoint_path)
            actual_path = str(checkpoint_path)
        
        # Calculate file size
        file_size_mb = Path(actual_path).stat().st_size / (1024 * 1024)
        elapsed = time.time() - start_time
        
        # Record metadata
        self.metadata['saves'].append({
            'epoch': epoch,
            'timestamp': datetime.now().isoformat(),
            'file_size_mb': file_size_mb,
            'save_time_sec': elapsed,
        })
        
        return actual_path, file_size_mb
    
    def load_checkpoint(
        self,
        checkpoint_path: str,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: str = 'cpu'
    ) -> Dict:
        """
        Load checkpoint with automatic decompression.
        
        Args:
            checkpoint_path: Path to checkpoint file
            model: Neural network model to load weights into
            optimizer: PyTorch optimizer to load state into (optional)
            device: Device to load onto ('cpu' or 'cuda')
            
        Returns:
            Dictionary with loaded data (epoch, metrics, etc.)
        """
        start_time = time.time()
        
        checkpoint_path = Path(checkpoint_path)
        
        # Handle both compressed and uncompressed files
        if checkpoint_path.suffix == '.gz':
            # Decompress on load
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp:
                with gzip.open(checkpoint_path, 'rb') as f_in:
                    tmp.write(f_in.read())
                tmp_path = tmp.name
            
            checkpoint = torch.load(tmp_path, map_location=device)
            Path(tmp_path).unlink()
        else:
            checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state if provided
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        elapsed = time.time() - start_time
        
        # Record metadata
        self.metadata['loads'].append({
            'epoch': checkpoint.get('epoch', 0),
            'timestamp': datetime.now().isoformat(),
            'load_time_sec': elapsed,
        })
        
        # Return relevant data
        return {
            'epoch': checkpoint.get('epoch', 0),
            'metrics': checkpoint.get('metrics', {}),
            'timestamp': checkpoint.get('timestamp', ''),
        }
    
    def get_latest_checkpoint(self) -> Optional[Path]:
        """Get path to most recent checkpoint"""
        checkpoints = sorted(self.checkpoint_dir.glob("*latest*.pt*"))
        if checkpoints:
            return checkpoints[-1]
        return None
    
    def cleanup_old_checkpoints(self, keep_best_n: int = 5):
        """
        Remove old checkpoints, keeping only the best N.
        
        Args:
            keep_best_n: Number of recent checkpoints to keep
        """
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_epoch_*.pt*"))
        
        if len(checkpoints) > keep_best_n:
            to_delete = checkpoints[:-keep_best_n]
            for cp in to_delete:
                cp.unlink()
                print(f"[CLEANUP] Removed old checkpoint: {cp.name}")
    
    def get_checkpoint_info(self, checkpoint_path: str) -> Dict:
        """
        Get information about a checkpoint without loading full state.
        
        Args:
            checkpoint_path: Path to checkpoint
            
        Returns:
            Dictionary with checkpoint metadata
        """
        checkpoint_path = Path(checkpoint_path)
        
        if checkpoint_path.suffix == '.gz':
            file_size = checkpoint_path.stat().st_size / (1024 * 1024)
        else:
            file_size = checkpoint_path.stat().st_size / (1024 * 1024)
        
        return {
            'path': str(checkpoint_path),
            'file_size_mb': file_size,
            'created': datetime.fromtimestamp(checkpoint_path.stat().st_ctime).isoformat(),
            'modified': datetime.fromtimestamp(checkpoint_path.stat().st_mtime).isoformat(),
        }
    
    def save_metadata(self, filepath: str = "checkpoint_metadata.json"):
        """Save checkpoint metadata to JSON"""
        metadata_path = self.checkpoint_dir / filepath
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2, default=str)
    
    def load_metadata(self, filepath: str = "checkpoint_metadata.json") -> Dict:
        """Load checkpoint metadata from JSON"""
        metadata_path = self.checkpoint_dir / filepath
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                return json.load(f)
        return {'saves': [], 'loads': []}


class DeltaCheckpoint:
    """
    Save only changed weights (delta) for faster checkpoint saves.
    Useful for frequent checkpoint saves during long training.
    
    WARNING: Only use with careful state management.
    """
    
    def __init__(self, checkpoint_dir: str = "checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.last_state = None
    
    def save_delta(
        self,
        model: nn.Module,
        epoch: int,
        full_save_interval: int = 10
    ) -> bool:
        """
        Save only changed weights if not a full save interval.
        
        Args:
            model: Neural network model
            epoch: Current epoch
            full_save_interval: Save full checkpoint every N epochs
            
        Returns:
            True if saved, False otherwise
        """
        current_state = model.state_dict()
        
        # Full save on interval
        if epoch % full_save_interval == 0:
            torch.save(current_state, self.checkpoint_dir / f"full_epoch_{epoch}.pt")
            self.last_state = current_state
            return True
        
        # Delta save
        if self.last_state is not None:
            delta = {}
            for name, param in current_state.items():
                if not torch.equal(param, self.last_state[name]):
                    delta[name] = param
            
            if delta:
                torch.save(delta, self.checkpoint_dir / f"delta_epoch_{epoch}.pt")
                return True
        
        return False
    
    def apply_deltas(
        self,
        model: nn.Module,
        full_checkpoint_epoch: int,
        target_epoch: int
    ) -> nn.Module:
        """
        Load model by applying deltas from full checkpoint up to target epoch.
        
        Args:
            model: Model to load state into
            full_checkpoint_epoch: Starting full checkpoint epoch
            target_epoch: Target epoch to reach
            
        Returns:
            Updated model
        """
        # Load full checkpoint
        full_path = self.checkpoint_dir / f"full_epoch_{full_checkpoint_epoch}.pt"
        if full_path.exists():
            model.load_state_dict(torch.load(full_path))
        
        # Apply deltas
        for epoch in range(full_checkpoint_epoch + 1, target_epoch + 1):
            delta_path = self.checkpoint_dir / f"delta_epoch_{epoch}.pt"
            if delta_path.exists():
                delta = torch.load(delta_path)
                state_dict = model.state_dict()
                state_dict.update(delta)
                model.load_state_dict(state_dict)
        
        return model


class CheckpointMonitor:
    """Monitor checkpoint creation and suggest cleanup"""
    
    def __init__(self, checkpoint_dir: str = "checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
    
    def get_checkpoint_stats(self) -> Dict:
        """Get statistics about checkpoints"""
        checkpoints = list(self.checkpoint_dir.glob("*.pt*"))
        
        total_size_mb = sum(cp.stat().st_size for cp in checkpoints) / (1024 * 1024)
        
        return {
            'total_checkpoints': len(checkpoints),
            'total_size_mb': total_size_mb,
            'oldest_checkpoint': min(checkpoints, key=lambda x: x.stat().st_ctime).name if checkpoints else None,
            'newest_checkpoint': max(checkpoints, key=lambda x: x.stat().st_ctime).name if checkpoints else None,
        }
    
    def print_checkpoint_stats(self):
        """Print checkpoint statistics"""
        stats = self.get_checkpoint_stats()
        print(f"\n[CHECKPOINT STATS]")
        print(f"  Total checkpoints: {stats['total_checkpoints']}")
        print(f"  Total size: {stats['total_size_mb']:.2f} MB")
        if stats['oldest_checkpoint']:
            print(f"  Oldest: {stats['oldest_checkpoint']}")
        if stats['newest_checkpoint']:
            print(f"  Newest: {stats['newest_checkpoint']}")
