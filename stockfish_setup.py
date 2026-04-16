import os
import requests
import zipfile
import io
import shutil

STOCKFISH_URL = "https://github.com/official-stockfish/Stockfish/releases/latest/download/stockfish-windows-x86-64-avx2.zip"
TARGET_DIR = os.path.join(os.path.dirname(__file__), "stockfish_bin")

def download_and_extract():
    if os.path.exists(TARGET_DIR):
        print(f"Stockfish directory already exists at {TARGET_DIR}")
        return

    os.makedirs(TARGET_DIR, exist_ok=True)
    print(f"Downloading Stockfish from {STOCKFISH_URL}...")
    
    try:
        response = requests.get(STOCKFISH_URL, stream=True)
        response.raise_for_status()
        
        print("Extracting Stockfish...")
        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
            zip_ref.extractall(TARGET_DIR)
            
        print(f"Stockfish setup complete. Executable is located inside: {TARGET_DIR}")
    except Exception as e:
        print(f"Failed to download and extract Stockfish: {e}")
        if os.path.exists(TARGET_DIR):
            shutil.rmtree(TARGET_DIR)

if __name__ == "__main__":
    download_and_extract()
