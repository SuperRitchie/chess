# Chess

Playable at [ritchiek.tech/chess](https://ritchiek.tech/chess/)

## AI modes

- **Random** chooses uniformly from legal moves
- **NN** batches the root policy and every child value, then uses them to guide a tactical negamax search
- **Neural MCTS** uses PUCT with batched leaf evaluation and virtual loss, followed by the same tactical verification within its two-second search budget

Both neural modes check for immediate mate before loading the model and consider every legal root move in tactical verification
The verifier uses iterative deepening, searches quiet replies as well as captures, and extends capture sequences at its leaves
An interrupted iteration is discarded so unchecked moves cannot beat completed evaluations
Neural values provide a bounded tie-break between close positional scores rather than overriding a forced mate or material loss

The policy/value network receives 18 feature planes: 12 piece planes, side to move, four castling-right planes, and the en-passant target. Its policy space represents normal moves plus queen, rook, bishop, and knight promotions separately.

## Continuous learning

Two serialized GitHub Actions workflows share the same model-training concurrency group:

1. **Nightly NN training** downloads recent Lichess games, samples positions, evaluates uncached positions with Stockfish, and trains the value head alongside the existing self-play policy data.
2. **Nightly policy-value self training** attempts 32 games in parallel batches, retains up to 60,000 replay samples, and trains both heads from `(state, MCTS visit distribution, final outcome)` samples

Each run can train on up to 24,000 self-play positions and 32,000 Stockfish positions

A candidate checkpoint replaces the current model only when:

- it improves the persistent holdout loss
- its MCTS alignment improves without a top-move accuracy regression
- when a compatible baseline exists, it meets the configured arena score and minimum number of decisive games

Holdout positions are excluded from training and accepted checkpoints preserve Adam optimizer state
Queued runs check out current master and staging rejects artifacts if inference code or accepted weights changed during training
Rejected candidates publish replay data and merged history without replacing accepted model files
Successful nightly workflows trigger deployment of current master so bot-generated model commits reach the playable site

## Local validation

```bash
pip install -r requirements.txt
python -m unittest discover -s ml/tests -p "test_*.py"

npm ci
CI=true npm test -- --watchAll=false
npm run build
```

## Measure browser engine changes

The benchmark uses the real TensorFlow.js model for both revisions and a Stockfish UCI executable to measure move quality
Run it from the modified checkout with a baseline worktree and keep the model files identical

```bash
git worktree add --detach ../chess-baseline a043cf96f9c25fb3b8971b3427f3dfe82a3c0ca5
python scripts/benchmark_engines.py --baseline-source ../chess-baseline --stockfish stockfish --positions 24 --game-pairs 1 --output engine-benchmark.json
```

The JSON records FENs, chosen moves, approximate centipawn loss, timings, model identity, and search diagnostics
Paired games swap the modified engine's color and save PGNs alongside the JSON
Games that reach the ply limit remain unfinished and are excluded from scored results
The harness runs TensorFlow.js on the Node CPU backend, so timings and search depth can differ from browser WebGL
