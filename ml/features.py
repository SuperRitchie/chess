# ml/features.py
"""
Chess features shared conceptually with src/ai/nnAI.js.

12 piece planes + side-to-move + four castling-right planes + one en-passant
square plane = 18 planes. These state features distinguish positions that have
the same piece placement but different legal moves.
"""
import chess
import numpy as np

PLANES = 18

PIECE_ORDER = [
    (chess.WHITE, chess.PAWN),
    (chess.WHITE, chess.KNIGHT),
    (chess.WHITE, chess.BISHOP),
    (chess.WHITE, chess.ROOK),
    (chess.WHITE, chess.QUEEN),
    (chess.WHITE, chess.KING),
    (chess.BLACK, chess.PAWN),
    (chess.BLACK, chess.KNIGHT),
    (chess.BLACK, chess.BISHOP),
    (chess.BLACK, chess.ROOK),
    (chess.BLACK, chess.QUEEN),
    (chess.BLACK, chess.KING),
]


def _square_to_row_col(square: chess.Square) -> tuple[int, int]:
    return 7 - chess.square_rank(square), chess.square_file(square)


def board_to_features(fen: str) -> np.ndarray:
    board = chess.Board(fen)
    planes: list[np.ndarray] = []

    for color, piece_type in PIECE_ORDER:
        mask = np.zeros((8, 8), dtype=np.float32)
        for square in board.pieces(piece_type, color):
            row, col = _square_to_row_col(square)
            mask[row, col] = 1.0
        planes.append(mask)

    planes.append(np.full((8, 8), 1.0 if board.turn == chess.WHITE else 0.0, dtype=np.float32))

    castling_rights = (
        board.has_kingside_castling_rights(chess.WHITE),
        board.has_queenside_castling_rights(chess.WHITE),
        board.has_kingside_castling_rights(chess.BLACK),
        board.has_queenside_castling_rights(chess.BLACK),
    )
    for available in castling_rights:
        planes.append(np.full((8, 8), 1.0 if available else 0.0, dtype=np.float32))

    en_passant = np.zeros((8, 8), dtype=np.float32)
    if board.ep_square is not None:
        row, col = _square_to_row_col(board.ep_square)
        en_passant[row, col] = 1.0
    planes.append(en_passant)

    features = np.stack(planes, axis=-1)
    if features.shape != (8, 8, PLANES):
        raise AssertionError(f"unexpected feature shape: {features.shape}")
    return features.reshape(-1).astype(np.float32)
