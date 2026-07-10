# ml/extract_positions.py
"""Read PGN and sample non-trivial positions into ml/data/positions.fen."""
import os
import pathlib
import random

import chess
import chess.pgn

IN_PGN = pathlib.Path("ml/data/games.pgn")
OUT_FEN = pathlib.Path("ml/data/positions.fen")
SEED = int(os.environ.get("POSITION_SEED", "42"))
random.seed(SEED)


def replay_to_ply(game, ply):
    board = game.board()
    for index, node in enumerate(game.mainline()):
        if index > ply:
            break
        if node.move is not None:
            board.push(node.move)
    return board


def sample_positions(pgn_path, max_games=2000, per_game=15, min_ply=12, max_ply=80):
    count = 0
    skipped = 0
    with open(pgn_path, "r", encoding="utf-8") as handle:
        while True:
            game = chess.pgn.read_game(handle)
            if game is None:
                break
            count += 1

            nodes = list(game.mainline())
            if not nodes:
                continue

            plies = [ply for ply in range(min(len(nodes), max_ply)) if ply >= min_ply]
            random.shuffle(plies)
            for ply in plies[:per_game]:
                try:
                    board = replay_to_ply(game, ply)
                    yield board.fen(en_passant="fen")
                except (AssertionError, ValueError, chess.IllegalMoveError) as exc:
                    skipped += 1
                    print(f"[extract] skipped malformed position in game {count}, ply {ply}: {exc}")
                    break

            if count >= max_games:
                break

    print(f"[extract] processed {count} games with seed {SEED}; skipped {skipped}")


def main():
    OUT_FEN.parent.mkdir(parents=True, exist_ok=True)
    positions = list(dict.fromkeys(sample_positions(IN_PGN)))
    if not positions:
        raise RuntimeError("no positions were extracted from the downloaded PGN")
    OUT_FEN.write_text("\n".join(positions) + "\n", encoding="utf-8")
    print(f"[extract] wrote {len(positions)} unique positions to {OUT_FEN}")


if __name__ == "__main__":
    main()
