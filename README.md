# Chess

Playable at [ritchiek.tech/chess](https://superritchie.github.io/chess/).

## AI modes

- **Random** chooses uniformly from legal moves.
- **NN** uses a shallow negamax search with the neural value head at leaf positions.
- **MCTS** is AlphaZero-style neural PUCT: policy priors guide selection and the neural value replaces random rollouts.

The policy/value network receives 18 feature planes: 12 piece planes, side to move, four castling-right planes, and the en-passant target. Its policy space represents normal moves plus queen, rook, bishop, and knight promotions separately.

## Continuous learning

Two serialized GitHub Actions workflows share the same model-training concurrency group:

1. **Nightly NN training** downloads recent Lichess games, samples positions, evaluates uncached positions with Stockfish, and trains the value head alongside the existing self-play policy data.
2. **Nightly policy-value self training** generates games with neural MCTS and trains both the policy and value heads from `(state, MCTS visit distribution, final outcome)` samples.

A candidate checkpoint replaces the current model only when:

- it improves on the persistent holdout set, and
- when a compatible baseline exists, it reaches the configured minimum score in a deterministic candidate-vs-baseline arena.

Holdout positions are excluded from training. Accepted checkpoints preserve Adam optimizer state. Generated-model commits are ignored by the push-triggered training workflow so they do not start recursive training runs.

## Local validation

```bash
pip install -r requirements.txt
python -m unittest discover -s ml/tests -p "test_*.py"

npm ci
CI=true npm test -- --watchAll=false
npm run build
```

## Useful training controls

The workflows provide defaults, and the main controls can also be overridden locally:

- `AZ_SELF_PLAY_GAMES`
- `AZ_MCTS_SEARCHES`
- `AZ_SELF_PLAY_SEED`
- `AZ_ARENA_GAMES`
- `AZ_ARENA_SEARCHES`
- `AZ_ARENA_MIN_SCORE`
- `AZ_MIN_VALIDATION_IMPROVEMENT`
- `SF_DEPTH`
