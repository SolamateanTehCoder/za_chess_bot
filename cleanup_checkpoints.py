"""Script to clean up old checkpoints while keeping the latest and final model."""

import os
import glob

CHECKPOINT_DIR = "checkpoints"

def cleanup_old_checkpoints():
    """
    Delete all checkpoint files except:
    - latest_checkpoint.pt (for resuming training)
    - final_model_100percent.pt (if exists - the final trained model)
    """
    
    if not os.path.exists(CHECKPOINT_DIR):
        print(f"Checkpoint directory '{CHECKPOINT_DIR}' does not exist.")
        return
    
    # Files to keep
    keep_files = {
        "latest_checkpoint.pt",
        "final_model_100percent.pt"
    }
    
    # Get all .pt files
    all_checkpoints = glob.glob(os.path.join(CHECKPOINT_DIR, "*.pt"))
    
    if not all_checkpoints:
        print("No checkpoint files found.")
        return
    
    deleted_count = 0
    kept_count = 0
    total_size_deleted = 0
    
    print(f"\nScanning {CHECKPOINT_DIR} directory...\n")
    
    for checkpoint_path in all_checkpoints:
        filename = os.path.basename(checkpoint_path)
        
        if filename in keep_files:
            kept_count += 1
            file_size = os.path.getsize(checkpoint_path) / (1024 * 1024)  # MB
            print(f"✓ Keeping: {filename} ({file_size:.2f} MB)")
        else:
            file_size = os.path.getsize(checkpoint_path) / (1024 * 1024)  # MB
            total_size_deleted += file_size
            os.remove(checkpoint_path)
            deleted_count += 1
            print(f"✗ Deleted: {filename} ({file_size:.2f} MB)")
    
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Files kept:    {kept_count}")
    print(f"  Files deleted: {deleted_count}")
    print(f"  Space freed:   {total_size_deleted:.2f} MB")
    print(f"{'='*60}\n")
    
    if kept_count > 0:
        print("Training can be resumed from 'latest_checkpoint.pt'")

if __name__ == "__main__":
    print("="*60)
    print("CHECKPOINT CLEANUP UTILITY")
    print("="*60)
    
    response = input("\nThis will delete all checkpoints except:\n"
                    "  - latest_checkpoint.pt\n"
                    "  - final_model_100percent.pt\n\n"
                    "Continue? (y/n): ").strip().lower()
    
    if response == 'y':
        cleanup_old_checkpoints()
        print("Cleanup complete!")
    else:
        print("Cleanup cancelled.")
