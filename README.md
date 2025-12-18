# Chess Blunder Prediction

This project uses Deep Learning (CNN) to predict the probability of a human player committing a "blunder" (a serious mistake) in a given chess position.

The model trains on real game data by analyzing the difference in engine evaluation (Stockfish) before and after a move.

## ⚠️ Important Data Note

**The original project relies on the massive Lichess Database.**

Since the full PGN database is hundreds of gigabytes and cannot be hosted on GitHub, this repository contains **only a small sample** (`data/raw_pgn/sample.pgn`) consisting of **10 games** for demonstration purposes.

To train the model effectively, you must:
1. Download a full PGN dump from [database.lichess.org](https://database.lichess.org/).
2. Place the file in the `data/raw_pgn/` directory.
3. Update the path in `dataset.py` to point to the full file.

## How It Works

The pipeline consists of three main stages:

### 1. Data Generation (`dataset.py`)
* **Parsing**: Reads games from PGN files.
* **Labeling**: Uses the Stockfish engine to evaluate positions.
    * Calculates the evaluation drop (delta) after a move.
    * If `eval_before - eval_after > 150` (centipawns), the move is labeled as a **Blunder (1)**.
    * Otherwise, it is a normal move (0).
* **Encoding**: Converts the board state into a **12x8x8 Spatial Tensor** (6 piece types × 2 colors) suitable for CNN input.

### 2. Model Architecture (`model.py`)
* **Type**: Convolutional Neural Network (CNN).
* **Layers**:
    * 2x Convolutional layers (Conv2d) to detect spatial patterns on the board.
    * Fully Connected (Linear) layers to process features.
    * **Sigmoid** activation output to represent the probability of a blunder ($P \in [0, 1]$).

### 3. Training & Evaluation
* **Training**: Optimized using Adam and Binary Cross Entropy Loss (`BCELoss`).
* **Evaluation**: Validated using **ROC-AUC** score to measure the model's ability to distinguish between safe moves and potential blunders.

## Installation & Usage

### 1. Prerequisites
Install the required Python packages:
```bash
pip install -r requirements.txt
```
Note: You must have the Stockfish engine installed and accessible in your system path for data generation.

2. Prepare Data
Run the processing script to generate NumPy arrays (X.npy, y.npy) from the PGN file:

Bash

python dataset.py
(By default, this processes the 10-game sample)

3. Train Model
Train the neural network:

Bash

python train.py
The trained model weights will be saved to model.pt.

4. Evaluate
Check the model's performance (ROC-AUC) on the processed data:

Bash

python evaluate.py
Project Structure
dataset.py: Handles PGN parsing, Stockfish interaction, and board tensor encoding.

model.py: PyTorch CNN architecture definition.

train.py: Training loop and optimization.

evaluate.py: Model inference and metric calculation.

data/: Directory for raw PGNs and processed .npy tensors.

