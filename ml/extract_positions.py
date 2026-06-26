# ml/extract_positions.py
"""
read PGN, sample midgame positions to avoid trivial openings/endings
outputs FEN lines to ml/data/positions.fen
"""
import pathlib
import random

import chess
import chess.pgn

IN_PGN = pathlib.Path("ml/data/games.pgn")
OUT_FEN = pathlib.Path("ml/data/positions.fen")
random.seed(42)


def replay_to_ply(game, ply):
    board = game.board()
    for i, node in enumerate(game.mainline()):
        if i > ply:
            break
        if node.move is not None:
            board.push(node.move)
    return board


def sample_positions(pgn_path, max_games=2000, per_game=15, min_ply=12, max_ply=80):
    count = 0
    skipped = 0
    with open(pgn_path, "r", encoding="utf-8") as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            count += 1

            nodes = list(game.mainline())
            if not nodes:
                continue

            plies = list(range(min(len(nodes), max_ply)))
            plies = [p for p in plies if p >= min_ply]
            random.shuffle(plies)
            plies = plies[:per_game]

            for ply in plies:
                try:
                    board = replay_to_ply(game, ply)
                    yield board.fen(en_passant="fen")
                except (AssertionError, ValueError, chess.IllegalMoveError) as exc:
                    skipped += 1
                    print(f"[extract] skipped malformed position in game {count}, ply {ply}: {exc}")
                    break

            if count >= max_games:
                break

    if skipped:
        print(f"[extract] skipped {skipped} malformed games or positions")


def main():
    OUT_FEN.parent.mkdir(parents=True, exist_ok=True)
    with OUT_FEN.open("w") as out:
        for fen in sample_positions(IN_PGN):
            out.write(fen + "\n")


if __name__ == "__main__":
    main()
