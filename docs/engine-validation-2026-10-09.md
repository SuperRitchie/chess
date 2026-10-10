# Engine validation — 2026-10-09

The revised engine uses the same accepted network weights as the baseline
The gains in this run come from search and move-selection changes

## Defects addressed

- sparse MCTS restricted tactical checks to the most visited root moves, excluding safer moves on small search budgets
- NN pruned root moves and replies using policy rankings and allowed unchecked candidates to compete against tactically checked scores
- tactical quiescence alone omitted quiet threats and did not provide a full minimax search
- legal move generation tested every destination square for every piece, consuming the search budget
- model commits made by the nightly bot did not trigger deployment and artifact staging could overwrite the other job’s history

## Reproduction

Baseline commit: `a043cf96f9c25fb3b8971b3427f3dfe82a3c0ca5`
Revised engine commit: `15af5108ce73a3bef1db7a8e3b31d3ef7ce9f750`
Model SHA-256: `2199d86c1f6112520a69c7001bc5f032adecb5ade13f532569bd8c011736bf92`
Baseline source SHA-256: `df03e987d1800710184850fa3447f88adf2495fb4ad108f801b7229290be74e5`
Revised source SHA-256: `9d825c6862f010043dfec35c045aa5c6b6f5469889b5d7f9d78a8efba711d085`

The harness loads the real TensorFlow.js model on the Node CPU backend and reconstructs pieces, castling rights, and en passant from FEN
Both revisions receive identical network weights
Position selection uses seed 20261009 and 24 Stockfish-source FENs from the persistent fixed evaluation set with an absolute stored score below 800 centipawns
The oracle is Stockfish 19 Lite WASM with 20000 search nodes, one thread, a 16 MiB hash table, and a cleared hash before each analysis
Centipawn loss compares the unrestricted oracle score with its score when restricted to the chosen move and clips negative differences to zero
The position run uses a 2000 ms MCTS budget and a 1200 ms NN tactical budget
The paired games use a 1000 ms MCTS budget and a 600 ms NN tactical budget, swapping the revised engine’s color after the same opening move 1 e4

```bash
git worktree add --detach ../chess-baseline a043cf96f9c25fb3b8971b3427f3dfe82a3c0ca5
npm ci
pip install -r requirements.txt
python scripts/benchmark_engines.py --baseline-source ../chess-baseline --stockfish "node /path/to/stockfish-19-lite-single.js" --positions 24 --oracle-nodes 20000 --game-pairs 1 --game-time-ms 1000 --max-plies 100 --output engine-benchmark.json
```

## Position results

| Mode | Metric | Baseline | Revised |
| --- | --- | ---: | ---: |
| MCTS | Mean centipawn loss | 98.54 | 41.25 |
| MCTS | Median centipawn loss | 71.00 | 24.00 |
| MCTS | Moves losing more than 200 cp | 3 | 0 |
| MCTS | Oracle top-move matches out of 24 | 5 | 8 |
| MCTS | Mean move time in ms | 1949 | 1948 |
| NN | Mean centipawn loss | 108.50 | 41.25 |
| NN | Median centipawn loss | 57.50 | 21.00 |
| NN | Moves losing more than 200 cp | 4 | 0 |
| NN | Oracle top-move matches out of 24 | 4 | 7 |
| NN | Mean move time in ms | 3778 | 1715 |

## Paired games

| Mode | Revised engine color | Result | Termination | Plies |
| --- | --- | --- | --- | ---: |
| MCTS | White | Win | checkmate | 56 |
| MCTS | Black | Win | checkmate | 47 |
| NN | White | Win | checkmate | 54 |
| NN | Black | Win | checkmate | 63 |

All four games ended in checkmate and every selected move was checked against python-chess legality
[Per-position and per-move records](engine-benchmark-2026-10-09.json) · [Game PGNs](engine-benchmark-2026-10-09.pgn)

## Checks and limits

21 JavaScript tests, 25 Python tests, and the production build pass locally
GitHub CI run 71 also passed the Python, browser, model optimization, build, and patch jobs
Regression coverage includes the starting-position perft totals 20, 400, and 8902, Kiwipete totals 48 and 2039, en passant pins, promotions, immediate mate, poisoned captures, budget interruptions, and stale/rejected training artifacts

This is a small development benchmark with one opening for the paired games and it does not establish an Elo rating
The fixed position set was used during development rather than reserved as an unseen engine-strength test
Node CPU timings and FEN-reconstructed state can differ from React game state and browser WebGL execution
The accepted model itself was not retrained in this change
