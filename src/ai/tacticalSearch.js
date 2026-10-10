import { isKingInCheck, listLegalMoves, makeMove } from '../rules/chessRules';
import { orderMoves, positionKey, positionScoreCp, scoreMove } from './chessHeuristics';

const MATE = 100000;
const ABORT = Symbol('search budget exhausted');
const opponent = (color) => color === 'white' ? 'black' : 'white';
export const searchMoveKey = (move) => `${move.from.x}-${move.from.y}:${move.to.x}-${move.to.y}:${move.promotionType || ''}`;

export function findMateInOne(pieces, color, enPassantTarget, moves = listLegalMoves(pieces, color, enPassantTarget)) {
  for (const move of moves) {
    const { pieces: next, nextEnPassant } = makeMove(
      pieces, move.from, move.to, move.promotionType || null, enPassantTarget,
    );
    const enemy = opponent(color);
    if (isKingInCheck(next, enemy) && listLegalMoves(next, enemy, nextEnPassant).length === 0) return move;
  }
  return null;
}

function visit(context) {
  if (context.nodes >= context.maxNodes || Date.now() >= context.deadline) throw ABORT;
  context.nodes += 1;
}

function quiescence(pieces, color, ep, alpha, beta, remaining, ply, context) {
  visit(context);
  const moves = listLegalMoves(pieces, color, ep);
  const inCheck = isKingInCheck(pieces, color);
  if (moves.length === 0) return inCheck ? -MATE + ply : 0;
  const standPat = positionScoreCp(pieces, color);
  if (remaining <= 0 && (!inCheck || remaining <= -2)) return standPat;

  let best = inCheck ? -Infinity : standPat;
  if (!inCheck) {
    if (best >= beta) return best;
    alpha = Math.max(alpha, best);
  }
  const forcing = inCheck ? moves : moves.filter((move) => move.capture || move.needsPromotion);
  for (const item of orderMoves(pieces, forcing, color, ep)) {
    const value = -quiescence(
      item.result.pieces, opponent(color), item.result.nextEnPassant,
      -beta, -alpha, remaining - 1, ply + 1, context,
    );
    best = Math.max(best, value);
    alpha = Math.max(alpha, best);
    if (alpha >= beta) break;
  }
  return best;
}

function negamax(pieces, color, ep, depth, alpha, beta, ply, context) {
  if (depth <= 0) return quiescence(pieces, color, ep, alpha, beta, context.quiescenceDepth, ply, context);
  visit(context);
  const cacheKey = `${positionKey(pieces, color, ep)}:${depth}:${ply}`;
  const cached = context.cache.get(cacheKey);
  if (cached) {
    if (cached.flag === 'exact') return cached.value;
    if (cached.flag === 'lower') alpha = Math.max(alpha, cached.value);
    if (cached.flag === 'upper') beta = Math.min(beta, cached.value);
    if (alpha >= beta) return cached.value;
  }
  const originalAlpha = alpha;
  const originalBeta = beta;
  const moves = listLegalMoves(pieces, color, ep);
  if (moves.length === 0) return isKingInCheck(pieces, color) ? -MATE + ply : 0;
  let best = -Infinity;
  let bestMove = null;
  for (const item of orderMoves(pieces, moves, color, ep, cached?.move)) {
    const value = -negamax(
      item.result.pieces, opponent(color), item.result.nextEnPassant,
      depth - 1, -beta, -alpha, ply + 1, context,
    );
    if (value > best) { best = value; bestMove = item.move; }
    alpha = Math.max(alpha, best);
    if (alpha >= beta) break;
  }
  const flag = best <= originalAlpha ? 'upper' : best >= originalBeta ? 'lower' : 'exact';
  context.cache.set(cacheKey, { value: best, flag, move: bestMove });
  return best;
}

export function searchTacticalMoves(pieces, color, enPassantTarget, {
  moves = listLegalMoves(pieces, color, enPassantTarget),
  neuralScores = new Map(),
  timeMs = 1000,
  maxDepth = 3,
  maxNodes = 20000,
  quiescenceDepth = 4,
  neuralTieBreakCp = 35,
} = {}) {
  if (moves.length === 0) return { move: null, depth: 0, nodes: 0 };
  const context = { deadline: Date.now() + Math.max(0, timeMs), maxNodes, quiescenceDepth, nodes: 0, cache: new Map() };
  const bonus = (move) => neuralTieBreakCp * Math.max(-1, Math.min(1, neuralScores.get(searchMoveKey(move)) || 0));
  let ranked = moves.map((move) => ({
    move,
    score: 650 * Math.atanh(Math.max(-0.999, Math.min(0.999, scoreMove(pieces, move, color, enPassantTarget)))),
  })).sort((a, b) => b.score + bonus(b.move) - a.score - bonus(a.move));
  let completedDepth = 0;

  for (let depth = 1; depth <= maxDepth; depth += 1) {
    const iteration = [];
    let bestAdjusted = -Infinity;
    try {
      for (const candidate of ranked) {
        const result = makeMove(pieces, candidate.move.from, candidate.move.to, candidate.move.promotionType || null, enPassantTarget);
        const rootAlpha = Math.abs(bestAdjusted) > MATE / 2 && Number.isFinite(bestAdjusted)
          ? bestAdjusted - Math.abs(neuralTieBreakCp)
          : bestAdjusted - bonus(candidate.move);
        const score = -negamax(result.pieces, opponent(color), result.nextEnPassant, depth - 1, -Infinity, -rootAlpha, 1, context);
        bestAdjusted = Math.max(bestAdjusted, Math.abs(score) > MATE / 2 ? score : score + bonus(candidate.move));
        iteration.push({ move: candidate.move, score });
      }
    } catch (error) {
      if (error !== ABORT) throw error;
      break;
    }
    // compare only complete iterations so timeouts never reward unchecked moves
    ranked = iteration.sort((a, b) => {
      if (Math.abs(a.score) > MATE / 2 || Math.abs(b.score) > MATE / 2) return b.score - a.score;
      return b.score + bonus(b.move) - a.score - bonus(a.move);
    });
    completedDepth = depth;
    if (ranked[0].score > MATE / 2) break;
  }
  return { move: ranked[0].move, scoreCp: ranked[0].score, depth: completedDepth, nodes: context.nodes };
}
