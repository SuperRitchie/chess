import * as tf from '@tensorflow/tfjs';
import { listLegalMoves, makeMove, getPiece } from '../rules/chessRules';

let modelPromise = null;

async function loadModel() {
  if (!modelPromise) {
    modelPromise = tf.loadLayersModel(`${process.env.PUBLIC_URL}/nn/model.json`);
  }
  return modelPromise;
}

function featuresFromBoard(pieces, isWhiteTurn) {
  const buf = tf.buffer([1, 8, 8, 13], 'float32');
  const typeToIdx = { pawn: 0, knight: 1, bishop: 2, rook: 3, queen: 4, king: 5 };

  for (let y = 0; y < 8; y++) {
    for (let x = 0; x < 8; x++) {
      const p = getPiece(pieces, x, y);
      if (p) {
        const base = p.color === 'white' ? 0 : 6;
        const c = base + typeToIdx[p.type];
        buf.set(1, 0, y, x, c);
      }
      buf.set(isWhiteTurn ? 1 : 0, 0, y, x, 12);
    }
  }

  return buf.toTensor();
}

async function evalPosition(model, pieces, isWhiteTurn) {
  const x = featuresFromBoard(pieces, isWhiteTurn);
  const y = model.predict(x);
  const val = (await y.data())[0];
  tf.dispose([x, y]);
  return val;
}

export async function pickNNMove(pieces, color, enPassantTarget, depth = 2) {
  const moves = listLegalMoves(pieces, color, enPassantTarget);
  let bestMove = null;
  let bestScore = -Infinity;

  for (const m of moves) {
    const promo = m.needsPromotion && !m.promotionType ? 'queen' : m.promotionType;
    const { pieces: after, nextEnPassant } = makeMove(pieces, m.from, m.to, promo, enPassantTarget);
    const nextColor = color === 'white' ? 'black' : 'white';
    const score = await minimax(after, depth - 1, nextColor, nextEnPassant);
    if (score > bestScore) {
      bestScore = score;
      bestMove = { ...m, promotionType: promo };
    }
  }

  return bestMove;
}

async function minimax(pieces, depth, color, enPassantTarget) {
  if (depth === 0) {
    const model = await loadModel();
    const isWhiteTurn = color === 'white';
    return evalPosition(model, pieces, isWhiteTurn);
  }

  const moves = listLegalMoves(pieces, color, enPassantTarget);
  if (moves.length === 0) {
    return color === 'white' ? -Infinity : Infinity;
  }

  const maximizing = color === 'white';
  let best = maximizing ? -Infinity : Infinity;
  const nextColor = color === 'white' ? 'black' : 'white';

  for (const m of moves) {
    const promo = m.needsPromotion && !m.promotionType ? 'queen' : m.promotionType;
    const { pieces: after, nextEnPassant } = makeMove(pieces, m.from, m.to, promo, enPassantTarget);
    const val = await minimax(after, depth - 1, nextColor, nextEnPassant);
    best = maximizing ? Math.max(best, val) : Math.min(best, val);
  }

  return best;
}
