// src/ai/randomAI.js
import { listLegalMoves } from './chessRules';

/**
 * Pick a random legal move for `color`.
 * Returns { from:{x,y}, to:{x,y}, promotionType?:'queen'|'rook'|'bishop'|'knight' }
 */
export async function pickRandomMove(pieces, color, enPassantTarget) {
    const moves = listLegalMoves(pieces, color, enPassantTarget);
    if (moves.length === 0) return null;

    // Always auto-queen when a promotion is possible and not specified
    const choice = moves[Math.floor(Math.random() * moves.length)];
    if (choice.needsPromotion && !choice.promotionType) {
        choice.promotionType = 'queen';
    }
    return choice;
}
