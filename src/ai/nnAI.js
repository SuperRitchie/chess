import * as tf from '@tensorflow/tfjs';
import { listLegalMoves, makeMove, getPiece } from '../rules/chessRules';

export const POLICY_SIZE = 64 * 64;

let modelPromise = null;

async function loadModel() {
  if (!modelPromise) {
    modelPromise = tf.loadLayersModel(`${process.env.PUBLIC_URL}/nn/model.json`);
  }
  return modelPromise;
}

function boardCoordToSquare(pos) {
  return pos.y + (7 - pos.x) * 8;
}

export function moveToPolicyIndex(move) {
  return boardCoordToSquare(move.from) * 64 + boardCoordToSquare(move.to);
}

function stableSoftmax(values) {
  const max = Math.max(...values);
  const exp = values.map((v) => Math.exp(v - max));
  const total = exp.reduce((sum, v) => sum + v, 0);
  if (!Number.isFinite(total) || total <= 0) {
    return values.map(() => 1 / Math.max(1, values.length));
  }
  return exp.map((v) => v / total);
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

async function rawPrediction(model, pieces, isWhiteTurn) {
  const x = featuresFromBoard(pieces, isWhiteTurn);
  const prediction = model.predict(x);
  const outputs = Array.isArray(prediction) ? prediction : [null, prediction];
  const [policyTensor, valueTensor] = outputs.length === 2 ? outputs : [null, outputs[0]];

  const policyData = policyTensor ? Array.from(await policyTensor.data()) : null;
  const valueData = await valueTensor.data();
  const value = valueData[0];

  const tensors = [x, ...outputs.filter(Boolean)];
  tf.dispose(tensors);

  return { policyData, value };
}

async function evalPosition(model, pieces, isWhiteTurn) {
  const { value } = await rawPrediction(model, pieces, isWhiteTurn);
  return value;
}

export async function predictPolicyValueForMoves(pieces, color, enPassantTarget) {
  const model = await loadModel();
  const legalMoves = listLegalMoves(pieces, color, enPassantTarget);
  if (legalMoves.length === 0) {
    return { value: -1, priors: new Map(), legalMoves };
  }

  const { policyData, value } = await rawPrediction(model, pieces, color === 'white');
  const priors = new Map();

  if (!policyData) {
    const uniform = 1 / legalMoves.length;
    legalMoves.forEach((move) => priors.set(moveToPolicyIndex(move), uniform));
    return { value, priors, legalMoves };
  }

  const logits = legalMoves.map((move) => policyData[moveToPolicyIndex(move)] ?? -1000000);
  const probs = stableSoftmax(logits);
  legalMoves.forEach((move, idx) => priors.set(moveToPolicyIndex(move), probs[idx]));

  return { value, priors, legalMoves };
}

export async function pickNNMove(pieces, color, enPassantTarget, depth = 2) {
  const moves = listLegalMoves(pieces, color, enPassantTarget);
  let bestMove = null;
  let bestScore = -Infinity;

  for (const m of moves) {
    const promo = m.needsPromotion && !m.promotionType ? 'queen' : m.promotionType;
    const { pieces: after, nextEnPassant } = makeMove(pieces, m.from, m.to, promo, enPassantTarget);
    const nextColor = color === 'white' ? 'black' : 'white';
    const score = -(await negamax(after, depth - 1, nextColor, nextEnPassant));
    if (score > bestScore) {
      bestScore = score;
      bestMove = { ...m, promotionType: promo };
    }
  }

  return bestMove;
}

async function negamax(pieces, depth, color, enPassantTarget) {
  const moves = listLegalMoves(pieces, color, enPassantTarget);
  if (moves.length === 0) {
    return -1;
  }

  if (depth === 0) {
    const model = await loadModel();
    return evalPosition(model, pieces, color === 'white');
  }

  const nextColor = color === 'white' ? 'black' : 'white';
  let best = -Infinity;

  for (const m of moves) {
    const promo = m.needsPromotion && !m.promotionType ? 'queen' : m.promotionType;
    const { pieces: after, nextEnPassant } = makeMove(pieces, m.from, m.to, promo, enPassantTarget);
    const val = -(await negamax(after, depth - 1, nextColor, nextEnPassant));
    best = Math.max(best, val);
  }

  return best;
}
