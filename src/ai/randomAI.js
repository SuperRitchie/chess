import { listLegalMoves } from '../rules/chessRules';

export async function pickRandomMove(pieces, color, enPassantTarget) {
  const moves = listLegalMoves(pieces, color, enPassantTarget);
  if (moves.length === 0) return null;

  const choice = moves[Math.floor(Math.random() * moves.length)];
  if (choice.needsPromotion && !choice.promotionType) {
    choice.promotionType = 'queen';
  }
  return choice;
}
