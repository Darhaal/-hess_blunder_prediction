## Predicting Chess Blunders with Machine Learning

This project models the probability of human chess blunders by learning from board positions and engine evaluations.

We generate a labeled dataset using Stockfish, encode board states as spatial tensors, and train a CNN to predict mistake likelihood.

Key aspects:
- Custom dataset generation from PGN
- Engine-based labeling
- Feature engineering for board representation
- Neural network evaluation with ROC-AUC
"# -hess_blunder_prediction" 
