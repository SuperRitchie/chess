# machine learning layout

```text
ml/
  checkpoints/          saved Keras model used as the next training starting point
  data/                 generated PGN, sampled positions, labels, and replay buffer
  fetch_lichess.py      downloads recent public games
  extract_positions.py  samples useful board positions from PGN
  stockfish_eval.py     labels positions with Stockfish centipawn scores
  train.py              continues training and exports the TensorFlow.js model to public/nn
  features.py           converts FEN boards into model inputs
  training_history.json nightly training metrics and resume status
```

## nightly learning flow

1. fetch recent Lichess games
2. sample board positions
3. label positions with Stockfish
4. merge new labels into `ml/data/replay_buffer.json`
5. load `ml/checkpoints/chess_eval.keras` when it exists
6. continue training from that saved brain
7. save the updated checkpoint, browser model, replay buffer, and metrics

The first run starts from scratch. Later runs continue from the saved checkpoint instead of replacing the model with a brand-new one.
