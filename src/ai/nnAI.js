import * as tf from '@tensorflow/tfjs';
import { listLegalMoves, makeMove, getPiece, isKingInCheck } from '../rules/chessRules';

export const POLICY_CHANNELS = 5;
export const POLICY_SIZE = 64 * 64 * POLICY_CHANNELS;
const LEGACY_POLICY_SIZE = 64 * 64;

let modelPromise = null;

async function loadModel() {
  if (!modelPromise) {
    modelPromise = tf.loadLayersModel(`${process.env.PUBLIC_URL}/nn/model.json`).catch((error) => {
      modelPromise = null;
      throw error;
    });
  }
  return modelPromise;
}

function boardCoordToSquare(pos) {
  return pos.y + (7 - pos.x) * 8;
}

function promotionChannel(move) {
  const promotion = move.promotionType || (move.needsPromotion ? 'queen' : null);
  return { knight: 1, bishop: 2, rook: 3, queen: 4 }[promotion] || 0;
}

export function moveToPolicyIndex(move, policySize = POLICY_SIZE) {
  const base = boardCoordToSquare(move.from) * 64 + boardCoordToSquare(move.to);
  return policySize === LEGACY_POLICY_SIZE ? base : base * POLICY_CHANNELS + promotionChannel(move);
}

export function moveKey(move) {
  return `${move.from.x}-${move.from.y}:${move.to.x}-${move.to.y}:${move.promotionType || ''}`;
}

function stableSoftmax(values) {
  const max = Math.max(...values);
  const exp = values.map((value) => Math.exp(value - max));
  const total = exp.reduce((sum, value) => sum + value, 0);
  if (!Number.isFinite(total) || total <= 0) {
    return values.map(() => 1 / Math.max(1, values.length));
  }
  return exp.map((value) => value / total);
}

function canCastleFromPieces(pieces, color, kingSide) {
  const row = color === 'white' ? 7 : 0;
  const king = getPiece(pieces, row, 4);
  const rook = getPiece(pieces, row, kingSide ? 7 : 0);
  return Boolean(
    king && king.type === 'king' && king.color === color && !king.hasMoved &&
    rook && rook.type === 'rook' && rook.color === color && !rook.hasMoved
  );
}

function featuresFromBoard(pieces, isWhiteTurn, enPassantTarget, planeCount = 18) {
  const buf = tf.buffer([1, 8, 8, planeCount], 'float32');
  const typeToIdx = { pawn: 0, knight: 1, bishop: 2, rook: 3, queen: 4, king: 5 };

  for (let y = 0; y < 8; y++) {
    for (let x = 0; x < 8; x++) {
      const piece = getPiece(pieces, y, x);
      if (piece) {
        const base = piece.color === 'white' ? 0 : 6;
        buf.set(1, 0, y, x, base + typeToIdx[piece.type]);
      }
      buf.set(isWhiteTurn ? 1 : 0, 0, y, x, 12);
    }
  }

  if (planeCount >= 18) {
    const rights = [
      canCastleFromPieces(pieces, 'white', true),
      canCastleFromPieces(pieces, 'white', false),
      canCastleFromPieces(pieces, 'black', true),
      canCastleFromPieces(pieces, 'black', false),
    ];
    rights.forEach((available, offset) => {
      if (!available) return;
      for (let y = 0; y < 8; y++) {
        for (let x = 0; x < 8; x++) buf.set(1, 0, y, x, 13 + offset);
      }
    });
    if (enPassantTarget) {
      buf.set(1, 0, enPassantTarget.x, enPassantTarget.y, 17);
    }
  }

  return buf.toTensor();
}

async function rawPrediction(model, pieces, isWhiteTurn, enPassantTarget) {
  const planeCount = Number(model.inputs?.[0]?.shape?.[3]) || 18;
  const x = featuresFromBoard(pieces, isWhiteTurn, enPassantTarget, planeCount);
  const prediction = model.predict(x);
  const outputs = Array.isArray(prediction) ? prediction : [null, prediction];
  const [policyTensor, valueTensor] = outputs.length === 2 ? outputs : [null, outputs[0]];

  try {
    const policyData = policyTensor ? Array.from(await policyTensor.data()) : null;
    const valueData = await valueTensor.data();
    return { policyData, value: valueData[0] };
  } finally {
    tf.dispose([x, ...outputs.filter(Boolean)]);
  }
}

async function evalPosition(model, pieces, isWhiteTurn, enPassantTarget) {
  const { value } = await rawPrediction(model, pieces, isWhiteTurn, enPassantTarget);
  return value;
}

function uniformPrediction(legalMoves) {
  const uniform = 1 / Math.max(1, legalMoves.length);
  const priors = new Map();
  legalMoves.forEach((move) => priors.set(moveKey(move), uniform));
  return { value: 0, priors, legalMoves };
}

export async function predictPolicyValueForMoves(pieces, color, enPassantTarget) {
  const legalMoves = listLegalMoves(pieces, color, enPassantTarget);
  if (legalMoves.length === 0) {
    return { value: isKingInCheck(pieces, color) ? -1 : 0, priors: new Map(), legalMoves };
  }

  try {
    const model = await loadModel();
    const { policyData, value } = await rawPrediction(model, pieces, color === 'white', enPassantTarget);
    if (!policyData) return { ...uniformPrediction(legalMoves), value };

    const logits = legalMoves.map((move) => policyData[moveToPolicyIndex(move, policyData.length)] ?? -1000000);
    const probabilities = stableSoftmax(logits);
    const priors = new Map();
    legalMoves.forEach((move, index) => priors.set(moveKey(move), probabilities[index]));
    return { value, priors, legalMoves };
  } catch (error) {
    console.warn('Neural model unavailable; using uniform policy/value fallback.', error);
    return uniformPrediction(legalMoves);
  }
}

export async function pickNNMove(pieces, color, enPassantTarget, depth = 2) {
  const moves = listLegalMoves(pieces, color, enPassantTarget);
  let bestMove = null;
  let bestScore = -Infinity;

  for (const move of moves) {
    const promotion = move.promotionType || (move.needsPromotion ? 'queen' : null);
    const { pieces: after, nextEnPassant } = makeMove(
      pieces,
      move.from,
      move.to,
      promotion,
      enPassantTarget,
    );
    const nextColor = color === 'white' ? 'black' : 'white';
    const score = -(await negamax(after, depth - 1, nextColor, nextEnPassant));
    if (score > bestScore) {
      bestScore = score;
      bestMove = { ...move, promotionType: promotion };
    }
  }

  return bestMove;
}

async function negamax(pieces, depth, color, enPassantTarget) {
  const moves = listLegalMoves(pieces, color, enPassantTarget);
  if (moves.length === 0) return isKingInCheck(pieces, color) ? -1 : 0;

  if (depth === 0) {
    try {
      const model = await loadModel();
      return await evalPosition(model, pieces, color === 'white', enPassantTarget);
    } catch (error) {
      return 0;
    }
  }

  const nextColor = color === 'white' ? 'black' : 'white';
  let best = -Infinity;
  for (const move of moves) {
    const promotion = move.promotionType || (move.needsPromotion ? 'queen' : null);
    const { pieces: after, nextEnPassant } = makeMove(
      pieces,
      move.from,
      move.to,
      promotion,
      enPassantTarget,
    );
    best = Math.max(best, -(await negamax(after, depth - 1, nextColor, nextEnPassant)));
  }
  return best;
}
