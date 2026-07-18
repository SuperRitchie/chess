import chess


def canonical_fen(fen: str) -> str:
    board = chess.Board(fen)
    fields = board.fen(en_passant="fen").split()
    return " ".join((*fields[:4], "0", "1"))
