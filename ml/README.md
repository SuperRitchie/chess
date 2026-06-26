# machine learning layout

```text
ml/
  data/                 generated PGN, sampled positions, and Stockfish labels
  fetch_lichess.py      downloads recent public games
  extract_positions.py  samples useful board positions from PGN
  stockfish_eval.py     labels positions with Stockfish centipawn scores
  train.py              trains and exports the TensorFlow.js model to public/nn
  features.py           converts FEN boards into model inputs
```

The nightly workflow runs these scripts in order and commits the exported browser model under `public/nn`.
