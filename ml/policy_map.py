# ml/policy_map.py
"""Stable chess move-to-policy mapping, including underpromotions."""
import chess

POLICY_VERSION = 2
POLICY_CHANNELS = 5
LEGACY_POLICY_SIZE = 64 * 64
POLICY_SIZE = LEGACY_POLICY_SIZE * POLICY_CHANNELS

_PROMOTION_TO_CHANNEL = {
    None: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
}
_CHANNEL_TO_PROMOTION = {value: key for key, value in _PROMOTION_TO_CHANNEL.items()}


def move_to_index(move: chess.Move) -> int:
    base = move.from_square * 64 + move.to_square
    channel = _PROMOTION_TO_CHANNEL.get(move.promotion)
    if channel is None:
        raise ValueError(f"unsupported promotion piece: {move.promotion}")
    return base * POLICY_CHANNELS + channel


def index_to_move(index: int, board: chess.Board | None = None) -> chess.Move:
    if not 0 <= index < POLICY_SIZE:
        raise ValueError(f"policy index out of range: {index}")

    base, channel = divmod(index, POLICY_CHANNELS)
    from_square = base // 64
    to_square = base % 64
    promotion = _CHANNEL_TO_PROMOTION[channel]

    # Channel zero is a normal move. For defensive compatibility, interpret an
    # old queen-promotion-shaped index as queen when a board makes that clear.
    if promotion is None and board is not None:
        piece = board.piece_at(from_square)
        if piece and piece.piece_type == chess.PAWN and chess.square_rank(to_square) in (0, 7):
            promotion = chess.QUEEN

    return chess.Move(from_square, to_square, promotion=promotion)


def normalize_policy_index(index: int, board: chess.Board | None, policy_version: int) -> int:
    """Convert stored legacy 4096-action policies into the version-2 mapping."""
    index = int(index)
    if policy_version >= POLICY_VERSION:
        if not 0 <= index < POLICY_SIZE:
            raise ValueError(f"policy index out of range: {index}")
        return index

    if not 0 <= index < LEGACY_POLICY_SIZE:
        raise ValueError(f"legacy policy index out of range: {index}")

    from_square = index // 64
    to_square = index % 64
    promotion = None
    if board is not None:
        piece = board.piece_at(from_square)
        if piece and piece.piece_type == chess.PAWN and chess.square_rank(to_square) in (0, 7):
            promotion = chess.QUEEN
    return move_to_index(chess.Move(from_square, to_square, promotion=promotion))
