"""Create a timelapse video from training progress plots."""

import os
import glob
import re
from datetime import datetime
import subprocess

PLOTS_DIR = "plots"
OUTPUT_DIR = "videos"
FPS = 24


def create_training_timelapse(fps=FPS):
    """
    Create an MP4 video from training progress plots.
    Plots are sorted chronologically and combined into a 24 FPS video.
    
    Args:
        fps: Frames per second for the output video
    """
    
    if not os.path.exists(PLOTS_DIR):
        print(f"Error: '{PLOTS_DIR}' directory does not exist.")
        return
    
    # Get all PNG files
    plot_files = glob.glob(os.path.join(PLOTS_DIR, "training_progress_*.png"))
    
    if len(plot_files) == 0:
        print(f"No plot files found in '{PLOTS_DIR}' directory.")
        return
    
    # Sort by timestamp in filename (format: training_progress_YYYYMMDD_HHMMSS.png)
    def extract_timestamp(filename):
        match = re.search(r'(\d{8}_\d{6})', filename)
        if match:
            return match.group(1)
        return ""
    
    plot_files.sort(key=extract_timestamp)
    
    print(f"\nFound {len(plot_files)} plot images")
    print(f"Creating timelapse at {fps} FPS...")
    print(f"Duration: {len(plot_files) / fps:.2f} seconds\n")
    
    # Create output directory
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    # Generate output filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(OUTPUT_DIR, f"training_timelapse_{timestamp}.mp4")
    
    # Create a temporary file list for ffmpeg
    filelist_path = os.path.join(OUTPUT_DIR, "filelist.txt")
    with open(filelist_path, 'w') as f:
        for plot_file in plot_files:
            # Use absolute path to avoid issues
            abs_path = os.path.abspath(plot_file).replace('\\', '/')
            # Duration each frame should be shown (1/fps seconds)
            duration = 1.0 / fps
            f.write(f"file '{abs_path}'\n")
            f.write(f"duration {duration}\n")
        # Add last frame again to ensure it's displayed
        if plot_files:
            abs_path = os.path.abspath(plot_files[-1]).replace('\\', '/')
            f.write(f"file '{abs_path}'\n")
    
    # Check if ffmpeg is available
    try:
        subprocess.run(['ffmpeg', '-version'], 
                      stdout=subprocess.DEVNULL, 
                      stderr=subprocess.DEVNULL, 
                      check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: ffmpeg is not installed or not in PATH")
        print("\nTo install ffmpeg:")
        print("  1. Download from: https://ffmpeg.org/download.html")
        print("  2. Or use: winget install ffmpeg")
        print("  3. Or use: choco install ffmpeg")
        os.remove(filelist_path)
        return
    
    # Create video using ffmpeg
    cmd = [
        'ffmpeg',
        '-f', 'concat',
        '-safe', '0',
        '-i', filelist_path,
        '-vf', f'fps={fps}',
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-crf', '18',  # High quality
        '-y',  # Overwrite output file
        output_file
    ]
    
    try:
        print("Running ffmpeg...")
        result = subprocess.run(cmd, 
                              capture_output=True, 
                              text=True, 
                              check=True)
        
        # Clean up temporary file
        os.remove(filelist_path)
        
        # Get output file size
        file_size = os.path.getsize(output_file) / (1024 * 1024)  # MB
        
        print(f"\n{'='*60}")
        print(f"✓ Video created successfully!")
        print(f"{'='*60}")
        print(f"Output: {output_file}")
        print(f"Frames: {len(plot_files)}")
        print(f"FPS: {fps}")
        print(f"Duration: {len(plot_files) / fps:.2f} seconds")
        print(f"Size: {file_size:.2f} MB")
        print(f"{'='*60}\n")
        
    except subprocess.CalledProcessError as e:
        print(f"Error creating video: {e}")
        print(f"FFmpeg output: {e.stderr}")
        if os.path.exists(filelist_path):
            os.remove(filelist_path)


def create_timelapse_with_custom_settings():
    """Interactive mode to customize video settings."""
    print("="*60)
    print("TRAINING TIMELAPSE VIDEO CREATOR")
    print("="*60 + "\n")
    
    # Check for plots
    plot_files = glob.glob(os.path.join(PLOTS_DIR, "training_progress_*.png"))
    
    if len(plot_files) == 0:
        print(f"No plot files found in '{PLOTS_DIR}' directory.")
        print("Run training with checkpoint saving enabled to generate plots.")
        return
    
    print(f"Found {len(plot_files)} plot images\n")
    
    # Get FPS
    fps_input = input(f"Enter FPS (default {FPS}): ").strip()
    fps = int(fps_input) if fps_input.isdigit() else FPS
    
    duration = len(plot_files) / fps
    print(f"\nVideo will be {duration:.2f} seconds at {fps} FPS\n")
    
    response = input("Create video? (y/n): ").strip().lower()
    
    if response == 'y':
        create_training_timelapse(fps=fps)
    else:
        print("Cancelled.")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--auto':
        # Automatic mode with default settings
        create_training_timelapse()
    else:
        # Interactive mode
        create_timelapse_with_custom_settings()
