# ml/policy_map.py
"""
fixed chess policy index mapping
"""
import chess

POLICY_SIZE = 64 * 64


def move_to_index(move: chess.Move) -> int:
    return move.from_square * 64 + move.to_square


def index_to_move(index: int, board: chess.Board | None = None) -> chess.Move:
    from_square = index // 64
    to_square = index % 64
    promotion = None

    if board is not None:
        piece = board.piece_at(from_square)
        if piece and piece.piece_type == chess.PAWN and chess.square_rank(to_square) in (0, 7):
            promotion = chess.QUEEN

    return chess.Move(from_square, to_square, promotion=promotion)


def to_queen_promotion(move: chess.Move) -> chess.Move:
    if move.promotion is not None and move.promotion != chess.QUEEN:
        return chess.Move(move.from_square, move.to_square, promotion=chess.QUEEN)
    return move
