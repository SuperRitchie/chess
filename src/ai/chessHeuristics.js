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

const PHASE_VALUES = {
  pawn: 0,
  knight: 1,
  bishop: 1,
  rook: 2,
  queen: 4,
  king: 0,
};

const TOTAL_PHASE = 24;

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

function piecePositionScore(pieces, piece, x, y, undeveloped, endgameWeight) {
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
    return score + advance * 18 * endgameWeight;
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
    let middleGameScore = 0;
    if (x === homeRow && (y === 2 || y === 6)) middleGameScore = 38;
    else if (piece.hasMoved && (x !== homeRow || y !== 4)) middleGameScore = -24;
    const endgameScore = centrality(x, y) * 9;
    return middleGameScore * (1 - endgameWeight) + endgameScore * endgameWeight;
  }

  return 0;
}

function pawnStructureScore(pieces, pawns, color, endgameWeight) {
  const enemy = opponent(color);
  const fileCounts = Array(8).fill(0);
  pawns.forEach(({ y }) => { fileCounts[y] += 1; });
  let score = 0;

  for (const pawn of pawns) {
    const hasNeighbor = (pawn.y > 0 && fileCounts[pawn.y - 1] > 0) ||
      (pawn.y < 7 && fileCounts[pawn.y + 1] > 0);
    if (!hasNeighbor) score -= 14;

    const direction = color === 'white' ? -1 : 1;
    let blockedByPawn = false;
    for (let x = pawn.x + direction; x >= 0 && x < 8 && !blockedByPawn; x += direction) {
      for (let y = Math.max(0, pawn.y - 1); y <= Math.min(7, pawn.y + 1); y += 1) {
        const piece = getPiece(pieces, x, y);
        if (piece?.type === 'pawn' && piece.color === enemy) {
          blockedByPawn = true;
          break;
        }
      }
    }

    if (!blockedByPawn) {
      const advance = color === 'white' ? 6 - pawn.x : pawn.x - 1;
      const passedBonuses = [0, 6, 14, 28, 48, 78, 125];
      score += passedBonuses[clamp(advance, 0, 6)] * (0.65 + 0.7 * endgameWeight);
      const supportRow = pawn.x - direction;
      const protectedByPawn = [pawn.y - 1, pawn.y + 1].some((file) => {
        const piece = getPiece(pieces, supportRow, file);
        return piece?.type === 'pawn' && piece.color === color;
      });
      if (protectedByPawn) score += 16;
    }
  }

  fileCounts.forEach((count) => {
    if (count > 1) score -= (count - 1) * 12;
  });
  return score;
}

function kingSafetyScore(pieces, king, color, endgameWeight) {
  if (!king) return 0;
  const direction = color === 'white' ? -1 : 1;
  const shieldRow = king.x + direction;
  let shield = 0;
  for (let file = Math.max(0, king.y - 1); file <= Math.min(7, king.y + 1); file += 1) {
    const piece = getPiece(pieces, shieldRow, file);
    if (piece?.type === 'pawn' && piece.color === color) shield += 1;
  }
  return (shield * 13 - (3 - shield) * 10) * (1 - endgameWeight);
}

function mopUpScore(kings, material, endgameWeight) {
  const advantage = material.white - material.black;
  if (Math.abs(advantage) < 180 || !kings.white || !kings.black) return 0;
  const winner = advantage > 0 ? 'white' : 'black';
  const loser = opponent(winner);
  const losingKing = kings[loser];
  const winningKing = kings[winner];
  const edgePressure = 7 - centrality(losingKing.x, losingKing.y);
  const kingDistance = Math.abs(losingKing.x - winningKing.x) + Math.abs(losingKing.y - winningKing.y);
  const score = (edgePressure * 8 + (14 - kingDistance) * 4) * endgameWeight;
  return winner === 'white' ? score : -score;
}

function positionScoreCp(pieces, color) {
  const undeveloped = {
    white: undevelopedMinorCount(pieces, 'white'),
    black: undevelopedMinorCount(pieces, 'black'),
  };
  const bishops = { white: 0, black: 0 };
  const pawns = { white: [], black: [] };
  const kings = { white: null, black: null };
  const material = { white: 0, black: 0 };
  const locatedPieces = [];
  let phase = 0;
  let whiteScore = 0;

  for (let x = 0; x < 8; x += 1) {
    for (let y = 0; y < 8; y += 1) {
      const piece = getPiece(pieces, x, y);
      if (!piece) continue;
      locatedPieces.push({ piece, x, y });
      phase += PHASE_VALUES[piece.type];
      material[piece.color] += PIECE_VALUES[piece.type];
      if (piece.type === 'bishop') bishops[piece.color] += 1;
      if (piece.type === 'pawn') pawns[piece.color].push({ x, y });
      if (piece.type === 'king') kings[piece.color] = { x, y };
    }
  }

  const endgameWeight = 1 - clamp(phase / TOTAL_PHASE, 0, 1);
  locatedPieces.forEach(({ piece, x, y }) => {
    const sign = piece.color === 'white' ? 1 : -1;
    whiteScore += sign * (
      PIECE_VALUES[piece.type] +
      piecePositionScore(pieces, piece, x, y, undeveloped[piece.color], endgameWeight)
    );
  });

  if (bishops.white >= 2) whiteScore += 18;
  if (bishops.black >= 2) whiteScore -= 18;
  whiteScore += pawnStructureScore(pieces, pawns.white, 'white', endgameWeight);
  whiteScore -= pawnStructureScore(pieces, pawns.black, 'black', endgameWeight);
  whiteScore += kingSafetyScore(pieces, kings.white, 'white', endgameWeight);
  whiteScore -= kingSafetyScore(pieces, kings.black, 'black', endgameWeight);
  whiteScore += mopUpScore(kings, material, endgameWeight);
  if (isKingInCheck(pieces, 'white')) whiteScore -= 45;
  if (isKingInCheck(pieces, 'black')) whiteScore += 45;

  return color === 'white' ? whiteScore : -whiteScore;
}

export function evaluatePosition(pieces, color) {
  return Math.tanh(positionScoreCp(pieces, color) / 650);
}

export function positionKey(pieces, color, enPassantTarget = null) {
  const board = Object.keys(pieces).sort().map((square) => {
    const piece = pieces[square];
    return `${square}:${piece.color}:${piece.type}:${piece.hasMoved ? 1 : 0}`;
  }).join('|');
  const enPassant = enPassantTarget ? `${enPassantTarget.x}-${enPassantTarget.y}` : '-';
  return `${color}:${enPassant}:${board}`;
}

function moveDetails(pieces, move, color, enPassantTarget) {
  const mover = getPiece(pieces, move.from.x, move.from.y);
  let captured = getPiece(pieces, move.to.x, move.to.y);
  if (!captured && mover?.type === 'pawn' && move.from.y !== move.to.y) {
    captured = getPiece(pieces, move.from.x, move.to.y);
  }
  const promotion = move.promotionType || (move.needsPromotion ? 'queen' : null);
  const result = makeMove(pieces, move.from, move.to, promotion, enPassantTarget);
  const givesCheck = isKingInCheck(result.pieces, opponent(color));
  const captureValue = captured ? PIECE_VALUES[captured.type] : 0;
  const moverValue = mover ? PIECE_VALUES[mover.type] : 0;
  let priority = captureValue * 12 - moverValue;
  if (promotion) priority += PIECE_VALUES[promotion] + 500;
  if (givesCheck) priority += 280;
  priority += positionScoreCp(result.pieces, color) * 0.02;
  return { move: { ...move, promotionType: promotion }, result, givesCheck, captureValue, priority };
}

export function orderMoves(pieces, moves, color, enPassantTarget = null, preferredMove = null) {
  const preferred = preferredMove ? `${preferredMove.from.x}-${preferredMove.from.y}:${preferredMove.to.x}-${preferredMove.to.y}:${preferredMove.promotionType || ''}` : null;
  return moves.map((move) => {
    const details = moveDetails(pieces, move, color, enPassantTarget);
    const id = `${details.move.from.x}-${details.move.from.y}:${details.move.to.x}-${details.move.to.y}:${details.move.promotionType || ''}`;
    return { ...details, priority: details.priority + (id === preferred ? 1000000 : 0) };
  }).sort((first, second) => second.priority - first.priority);
}

function quiescence(pieces, color, enPassantTarget, depth, alpha, beta, extensionBudget, context) {
  if (Date.now() >= context.deadline || context.nodes >= context.maxNodes) {
    return evaluatePosition(pieces, color);
  }
  context.nodes += 1;

  const cacheKey = `${positionKey(pieces, color, enPassantTarget)}:${depth}:${extensionBudget}`;
  const cached = context.cache.get(cacheKey);
  if (cached !== undefined) return cached;

  const legalMoves = listLegalMoves(pieces, color, enPassantTarget);
  if (legalMoves.length === 0) return isKingInCheck(pieces, color) ? -1 : 0;

  const inCheck = isKingInCheck(pieces, color);
  const standPat = evaluatePosition(pieces, color);
  if (depth <= 0 && (!inCheck || extensionBudget <= 0)) return standPat;

  let best = inCheck ? -1 : standPat;
  if (!inCheck) {
    if (best >= beta) return best;
    alpha = Math.max(alpha, best);
  }

  let ordered = orderMoves(pieces, legalMoves, color, enPassantTarget);
  if (!inCheck) {
    ordered = ordered.filter((item) => item.captureValue > 0 || item.move.promotionType || item.givesCheck);
    ordered = ordered.slice(0, context.maxMoves);
  }
  if (ordered.length === 0) return standPat;

  let completed = true;
  for (const item of ordered) {
    const nextExtensionBudget = inCheck && depth <= 0 ? extensionBudget - 1 : extensionBudget;
    const score = -quiescence(
      item.result.pieces,
      opponent(color),
      item.result.nextEnPassant,
      depth - 1,
      -beta,
      -alpha,
      nextExtensionBudget,
      context,
    );
    best = Math.max(best, score);
    alpha = Math.max(alpha, best);
    if (alpha >= beta || Date.now() >= context.deadline || context.nodes >= context.maxNodes) {
      completed = false;
      break;
    }
  }

  if (completed) context.cache.set(cacheKey, best);
  return best;
}

export function tacticalSearch(
  pieces,
  color,
  enPassantTarget = null,
  {
    depth = 2,
    extensionBudget = 1,
    maxMoves = 6,
    maxNodes = 600,
    deadline = Infinity,
    cache = new Map(),
  } = {},
) {
  const context = { cache, deadline, maxMoves, maxNodes, nodes: 0 };
  return quiescence(
    pieces,
    color,
    enPassantTarget,
    Math.max(0, depth),
    -1,
    1,
    Math.max(0, extensionBudget),
    context,
  );
}

export function tacticalScoreForMove(
  pieces,
  move,
  color,
  enPassantTarget = null,
  options = {},
) {
  const details = moveDetails(pieces, move, color, enPassantTarget);
  return -tacticalSearch(
    details.result.pieces,
    opponent(color),
    details.result.nextEnPassant,
    options,
  );
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
