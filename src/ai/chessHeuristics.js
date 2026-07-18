import {
  getPiece,
  isKingInCheck,
  isSquareAttacked,
  listLegalMoves,
  makeMove,
} from '../rules/chessRules';

const PIECE_VALUES = {
  pawn: 100,
  knight: 320,
  bishop: 330,
  rook: 500,
  queen: 900,
  king: 0,
};

const clamp = (value, minimum, maximum) => Math.max(minimum, Math.min(maximum, value));
const opponent = (color) => (color === 'white' ? 'black' : 'white');
const key = (x, y) => `${x}-${y}`;

function centrality(x, y) {
  return 7 - Math.abs(x - 3.5) - Math.abs(y - 3.5);
}

function undevelopedMinorCount(pieces, color) {
  const row = color === 'white' ? 7 : 0;
  return [1, 2, 5, 6].filter((column) => {
    const piece = getPiece(pieces, row, column);
    return piece && (piece.type === 'knight' || piece.type === 'bishop') && !piece.hasMoved;
  }).length;
}

function piecePositionScore(pieces, piece, x, y, undeveloped) {
  const homeRow = piece.color === 'white' ? 7 : 0;
  const pawnRow = piece.color === 'white' ? 6 : 1;
  const advance = Math.abs(pawnRow - x);

  if (piece.type === 'pawn') {
    let score = advance * 7;
    if (y === 3 || y === 4) score += 12;
    else if (y === 2 || y === 5) score += 4;
    if ((y <= 1 || y >= 6) && undeveloped >= 2) score -= advance * 18;

    const king = getPiece(pieces, homeRow, 4);
    if (y === 5 && advance > 0 && king?.type === 'king' && !king.hasMoved) {
      score -= advance * 28;
    }
    return score;
  }

  if (piece.type === 'knight') {
    const homeSquare = x === homeRow && (y === 1 || y === 6);
    return centrality(x, y) * 7 - (homeSquare && !piece.hasMoved ? 12 : 0);
  }

  if (piece.type === 'bishop') {
    const homeSquare = x === homeRow && (y === 2 || y === 5);
    return centrality(x, y) * 3 - (homeSquare && !piece.hasMoved ? 8 : 0);
  }

  if (piece.type === 'rook') {
    return centrality(x, y) * 0.5;
  }

  if (piece.type === 'queen') {
    return centrality(x, y) - (piece.hasMoved && undeveloped >= 3 ? 18 : 0);
  }

  if (piece.type === 'king') {
    if (x === homeRow && (y === 2 || y === 6)) return 38;
    if (piece.hasMoved && (x !== homeRow || y !== 4)) return -24;
  }

  return 0;
}

function positionScoreCp(pieces, color) {
  const undeveloped = {
    white: undevelopedMinorCount(pieces, 'white'),
    black: undevelopedMinorCount(pieces, 'black'),
  };
  const bishops = { white: 0, black: 0 };
  let whiteScore = 0;

  for (let x = 0; x < 8; x += 1) {
    for (let y = 0; y < 8; y += 1) {
      const piece = getPiece(pieces, x, y);
      if (!piece) continue;
      const sign = piece.color === 'white' ? 1 : -1;
      whiteScore += sign * (
        PIECE_VALUES[piece.type] +
        piecePositionScore(pieces, piece, x, y, undeveloped[piece.color])
      );
      if (piece.type === 'bishop') bishops[piece.color] += 1;
    }
  }

  if (bishops.white >= 2) whiteScore += 18;
  if (bishops.black >= 2) whiteScore -= 18;
  if (isKingInCheck(pieces, 'white')) whiteScore -= 45;
  if (isKingInCheck(pieces, 'black')) whiteScore += 45;

  return color === 'white' ? whiteScore : -whiteScore;
}

export function evaluatePosition(pieces, color) {
  return Math.tanh(positionScoreCp(pieces, color) / 650);
}

export function scoreMove(pieces, move, color, enPassantTarget = null) {
  const promotion = move.promotionType || (move.needsPromotion ? 'queen' : null);
  const { pieces: after, nextEnPassant } = makeMove(
    pieces,
    move.from,
    move.to,
    promotion,
    enPassantTarget,
  );
  const enemy = opponent(color);
  const givesCheck = isKingInCheck(after, enemy);
  if (givesCheck && listLegalMoves(after, enemy, nextEnPassant).length === 0) return 1;

  let scoreCp = positionScoreCp(after, color);
  const movedPiece = getPiece(after, move.to.x, move.to.y);
  if (movedPiece && isSquareAttacked(after, move.to.x, move.to.y, enemy)) {
    const withoutMovedPiece = { ...after };
    delete withoutMovedPiece[key(move.to.x, move.to.y)];
    const defended = isSquareAttacked(
      withoutMovedPiece,
      move.to.x,
      move.to.y,
      color,
    );
    scoreCp -= PIECE_VALUES[movedPiece.type] * (defended ? 0.08 : 0.55);
  }
  if (givesCheck) scoreCp += 24;

  return Math.tanh(scoreCp / 650);
}

function stableSoftmax(values) {
  const max = Math.max(...values);
  const exponentials = values.map((value) => Math.exp(value - max));
  const total = exponentials.reduce((sum, value) => sum + value, 0);
  if (!Number.isFinite(total) || total <= 0) {
    return values.map(() => 1 / Math.max(1, values.length));
  }
  return exponentials.map((value) => value / total);
}

export function stabilizePolicyValue(
  pieces,
  color,
  enPassantTarget,
  prediction,
  { neuralValueWeight = 0.55, neuralPolicyWeight = 0.55 } = {},
) {
  const legalMoves = prediction?.legalMoves || listLegalMoves(pieces, color, enPassantTarget);
  if (legalMoves.length === 0) {
    return {
      value: isKingInCheck(pieces, color) ? -1 : 0,
      legalMoves,
      priors: new Map(),
      moveScores: new Map(),
    };
  }

  const moveScores = new Map();
  const heuristicScores = legalMoves.map((move) => {
    const score = scoreMove(pieces, move, color, enPassantTarget);
    moveScores.set(`${move.from.x}-${move.from.y}:${move.to.x}-${move.to.y}:${move.promotionType || ''}`, score);
    return score * 8;
  });
  const heuristicPriors = stableSoftmax(heuristicScores);
  const neuralAvailable = prediction?.neuralAvailable !== false;
  const neuralTotal = legalMoves.reduce((sum, move) => {
    const probability = Number(prediction?.priors?.get(
      `${move.from.x}-${move.from.y}:${move.to.x}-${move.to.y}:${move.promotionType || ''}`,
    ));
    return sum + (Number.isFinite(probability) && probability > 0 ? probability : 0);
  }, 0);
  const policyWeight = neuralAvailable && neuralTotal > 0
    ? clamp(neuralPolicyWeight, 0, 1)
    : 0;
  const priors = new Map();

  legalMoves.forEach((move, index) => {
    const moveId = `${move.from.x}-${move.from.y}:${move.to.x}-${move.to.y}:${move.promotionType || ''}`;
    const rawNeuralPrior = Number(prediction?.priors?.get(moveId)) || 0;
    const neuralPrior = rawNeuralPrior > 0 ? rawNeuralPrior / neuralTotal : 0;
    priors.set(
      moveId,
      policyWeight * neuralPrior + (1 - policyWeight) * heuristicPriors[index],
    );
  });

  const staticValue = evaluatePosition(pieces, color);
  const valueWeight = neuralAvailable ? clamp(neuralValueWeight, 0, 1) : 0;
  const neuralValue = Number.isFinite(prediction?.value) ? prediction.value : 0;
  const value = clamp(valueWeight * neuralValue + (1 - valueWeight) * staticValue, -1, 1);

  return { ...prediction, value, legalMoves, priors, moveScores };
}

export function rankMoves(prediction) {
  return [...prediction.legalMoves].sort((first, second) => {
    const firstId = `${first.from.x}-${first.from.y}:${first.to.x}-${first.to.y}:${first.promotionType || ''}`;
    const secondId = `${second.from.x}-${second.from.y}:${second.to.x}-${second.to.y}:${second.promotionType || ''}`;
    const firstScore = (prediction.priors.get(firstId) || 0) + 0.35 * (prediction.moveScores.get(firstId) || 0);
    const secondScore = (prediction.priors.get(secondId) || 0) + 0.35 * (prediction.moveScores.get(secondId) || 0);
    return secondScore - firstScore;
  });
}
