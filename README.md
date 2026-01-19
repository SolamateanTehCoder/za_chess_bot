# ♟️ ZA Chess Bot

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)

**ZA Chess Bot** is a high-performance chess engine built in Python. This project is driven by a single, ambitious goal: to climb the ranks of computer chess championships and eventually challenge the dominance of engines like Stockfish.

We aren't just making another engine; we're building a contender.

## 🚀 The Vision
Most engines are content being "strong." ZA Chess Bot is built for the **Computer Chess Championship (CCC)** and **TCEC** environments. We are focused on:
- **Aggressive Search Optimization:** Implementing advanced pruning and move ordering.
- **Neural Evaluation:** Moving beyond static heuristics to deep-learning-based board evaluation.
- **Efficiency:** Squeezing every bit of performance out of Python (and future C++ extensions).

## ✨ Features (Current & Incoming)
- [x] **UCI Protocol Support:** Compatible with popular GUIs like Arena, Fritz, and Cutechess.
- [x] **Minimax with Alpha-Beta Pruning:** The core search foundation.
- [ ] **Iterative Deepening:** Dynamic depth management for time control.
- [ ] **Transposition Tables:** Zobrist hashing to prevent re-calculating positions.
- [ ] **NNUE Integration:** Efficiently Updatable Neural Networks for god-tier evaluation.
- [ ] **Quiescence Search:** Eliminating the "horizon effect" during captures.

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/SolamateanTehCoder/za_chess_bot.git](https://github.com/SolamateanTehCoder/za_chess_bot.git)
   cd za_chess_bot
