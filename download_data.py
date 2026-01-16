"""
Script to download and parse high-quality chess games for training.
Downloads PGN files from the web, parses them, and converts to JSONL.
"""

import requests
import chess.pgn
from pathlib import Path
import json
from tqdm import tqdm
import zipfile
import io

def download_and_parse_pgn(url: str, output_file: str, max_games: int = 10000):
    """
    Download PGN file (or zip archive), parse games, and save as JSONL.

    Args:
        url: URL to PGN file or .zip archive containing a single .pgn
        output_file: Path to save JSONL file
        max_games: Maximum games to process
    """
    try:
        print(f"Downloading from {url}...")
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        pgn_content = ""
        if url.endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                # Find the PGN file in the zip archive
                for filename in z.namelist():
                    if filename.endswith(".pgn"):
                        with z.open(filename) as pgn_file:
                            pgn_content = pgn_file.read().decode("latin-1")
                        break
        else:
            pgn_content = response.text

        if not pgn_content:
            print("No PGN content found.")
            return

        print("Parsing PGN and converting to JSONL...")
        games_processed = 0
        pgn_io = io.StringIO(pgn_content)

        with open(output_file, "w") as jsonl_file:
            with tqdm(total=max_games) as pbar:
                while True:
                    if games_processed >= max_games:
                        break

                    game = chess.pgn.read_game(pgn_io)
                    if game is None:
                        break

                    moves = [move.uci() for move in game.mainline_moves()]
                    result = game.headers.get("Result", "*")

                    if moves and result != "*":
                        game_data = {
                            "moves": moves,
                            "result": result,
                            "white_elo": game.headers.get("WhiteElo", "N/A"),
                            "black_elo": game.headers.get("BlackElo", "N/A"),
                            "event": game.headers.get("Event", "N/A"),
                        }
                        jsonl_file.write(json.dumps(game_data) + "\n")
                        games_processed += 1
                        pbar.update(1)

        print(f"Successfully processed {games_processed} games.")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download and parse chess PGN files.")
    parser.add_argument("--url", type=str, required=True, help="URL of the PGN file to download.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output JSONL file.")
    parser.add_argument("--max_games", type=int, default=10000, help="Maximum number of games to process.")

    args = parser.parse_args()

    download_and_parse_pgn(args.url, args.output_file, max_games=args.max_games)
