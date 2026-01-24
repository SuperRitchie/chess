# ml/extract_positions.py
"""
Read PGN, sample midgame positions to avoid trivial openings/endings.
Outputs FEN lines to ml/data/positions.fen
"""
import pathlib, random
import chess.pgn

IN_PGN = pathlib.Path("ml/data/games.pgn")
OUT_FEN = pathlib.Path("ml/data/positions.fen")
random.seed(42)

def sample_positions(pgn_path, max_games=10000, per_game=15, min_ply=12, max_ply=80):
    count = 0
    with open(pgn_path, "r", encoding="utf-8") as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None: break
            count += 1
            board = game.board()
            nodes = list(game.mainline())
            if not nodes: continue

            plies = list(range(min(len(nodes), max_ply)))
            plies = [p for p in plies if p >= min_ply]
            random.shuffle(plies)
            plies = plies[:per_game]

            for ply in plies:
                board.reset()
                for i, node in enumerate(game.mainline()):
                    if i > ply: break
                    if node.move is not None:
                        board.push(node.move)
                yield board.fen(en_passant="fen")

            if count >= max_games:
                break

def main():
    OUT_FEN.parent.mkdir(parents=True, exist_ok=True)
    with OUT_FEN.open("w") as out:
        for fen in sample_positions(IN_PGN):
            out.write(fen + "\n")

if __name__ == "__main__":
    main()
