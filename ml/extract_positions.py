# ml/extract_positions.py
"""sample new non-trivial positions from PGN data"""
import json
import os
import pathlib
import random

import chess
import chess.pgn

from fen_utils import canonical_fen

IN_PGN = pathlib.Path("ml/data/games.pgn")
OUT_FEN = pathlib.Path("ml/data/positions.fen")
REPLAY_JSON = pathlib.Path("ml/data/replay_buffer.json")
SEED = int(os.environ.get("POSITION_SEED", "42"))
TARGET_NEW = int(os.environ.get("POSITION_TARGET_NEW", "4000"))
MIN_NEW = int(os.environ.get("POSITION_MIN_NEW", "1000"))
MAX_GAMES = int(os.environ.get("POSITION_MAX_GAMES", "4000"))
PER_GAME = int(os.environ.get("POSITION_PER_GAME", "32"))
random.seed(SEED)


def sample_positions(pgn_path, max_games=MAX_GAMES, per_game=PER_GAME, min_ply=12, max_ply=100):
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
            selected = set(plies[:per_game])
            board = game.board()
            for ply, node in enumerate(nodes):
                try:
                    if node.move is not None:
                        board.push(node.move)
                    if ply in selected:
                        yield canonical_fen(board.fen(en_passant="fen"))
                except (AssertionError, ValueError, chess.IllegalMoveError) as exc:
                    skipped += 1
                    print(f"[extract] skipped malformed position in game {count}, ply {ply}: {exc}")
                    break

            if count >= max_games:
                break

    print(f"[extract] processed {count} games with seed {SEED}; skipped {skipped}")


def read_seen_fens() -> set[str]:
    if not REPLAY_JSON.exists():
        return set()
    try:
        items = json.loads(REPLAY_JSON.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return set()

    seen = set()
    for item in items if isinstance(items, list) else []:
        fen = item.get("fen")
        if not fen:
            continue
        try:
            seen.add(canonical_fen(fen))
        except ValueError:
            continue
    return seen


def main():
    OUT_FEN.parent.mkdir(parents=True, exist_ok=True)
    positions = list(dict.fromkeys(sample_positions(IN_PGN)))
    if not positions:
        raise RuntimeError("no positions were extracted from the downloaded PGN")

    seen = read_seen_fens()
    unseen = [fen for fen in positions if fen not in seen]
    random.shuffle(unseen)
    selected = unseen[:TARGET_NEW]
    if len(selected) < MIN_NEW:
        raise RuntimeError(
            f"only {len(selected)} new positions found, below POSITION_MIN_NEW={MIN_NEW}"
        )

    OUT_FEN.write_text("\n".join(selected) + "\n", encoding="utf-8")
    print(
        f"[extract] candidates {len(positions)}, already seen {len(positions) - len(unseen)}, "
        f"wrote {len(selected)} new positions to {OUT_FEN}"
    )


if __name__ == "__main__":
    main()
