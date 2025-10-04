# ml/features.py
"""
Feature extraction consistent with src/ai/nnAI.js:
12 piece planes (6 per color) + 1 side-to-move plane -> 8*8*13 = 832 features.
"""
import numpy as np
import chess

piece_order = [
    (chess.WHITE, chess.PAWN), (chess.WHITE, chess.KNIGHT), (chess.WHITE, chess.BISHOP),
    (chess.WHITE, chess.ROOK), (chess.WHITE, chess.QUEEN), (chess.WHITE, chess.KING),
    (chess.BLACK, chess.PAWN), (chess.BLACK, chess.KNIGHT), (chess.BLACK, chess.BISHOP),
    (chess.BLACK, chess.ROOK), (chess.BLACK, chess.QUEEN), (chess.BLACK, chess.KING),
]

def board_to_features(fen):
    board = chess.Board(fen)
    planes = []
    for color, ptype in piece_order:
        mask = np.zeros((8,8), dtype=np.float32)
        for sq in board.pieces(ptype, color):
            r = 7 - (sq // 8)  # rank 8..1 -> 0..7 top-down
            c = sq % 8
            mask[r, c] = 1.0
        planes.append(mask)
    side = np.full((8,8), 1.0 if board.turn == chess.WHITE else 0.0, dtype=np.float32)
    planes.append(side)
    x = np.stack(planes, axis=-1)   # [8,8,13]
    return x.reshape(-1).astype(np.float32)  # 832,
