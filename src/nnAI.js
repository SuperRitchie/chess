// src/nnAI.js
import * as tf from "@tensorflow/tfjs";
import { listLegalMoves, makeMove, getPiece } from "./chessRules";

/**
 * Load the NN model once (singleton).
 */
let _modelPromise = null;

async function loadModel() {
  if (!_modelPromise) {
    const url = process.env.PUBLIC_URL + "/nn/model.json";
    console.log("Loading NN model from", url);
    _modelPromise = tf.loadLayersModel(url);
  }
  return _modelPromise;
}

// Channel indices must match ml/features.py::piece_order
// piece_order = [W pawn, W knight, W bishop, W rook, W queen, W king,
//                B pawn, B knight, B bishop, B rook, B queen, B king] + side plane
const pieceChannels = {
  white: {
    pawn: 0,
    knight: 1,
    bishop: 2,
    rook: 3,
    queen: 4,
    king: 5,
  },
  black: {
    pawn: 6,
    knight: 7,
    bishop: 8,
    rook: 9,
    queen: 10,
    king: 11,
  },
};

/**
 * Build a tensor of shape [1, 8, 8, 13] matching the Python feature extractor.
 */
function featuresFromBoard(pieces, isWhiteTurn) {
  const data = [];

  // x = row (0..7 top to bottom), y = col (0..7 left to right)
  // We push per-square 13-dim feature vectors; channel axis is last.
  for (let x = 0; x < 8; x++) {
    for (let y = 0; y < 8; y++) {
      const cell = new Array(13).fill(0);
      const p = getPiece(pieces, x, y);

      if (p) {
        const ch = pieceChannels[p.color][p.type];
        cell[ch] = 1;
      }

      // side-to-move plane (same for all squares)
      cell[12] = isWhiteTurn ? 1 : 0;

      // Push channels-last so flatten order matches [8,8,13] in NumPy
      data.push(...cell);
    }
  }

  // [1, 8, 8, 13]: batch, row, col, channel
  return tf.tensor4d(data, [1, 8, 8, 13]);
}

/**
 * Score: predicted centipawns from White's perspective.
 */
async function evalPosition(model, pieces, isWhiteTurn) {
  const x = featuresFromBoard(pieces, isWhiteTurn);
  const y = model.predict(x);
  const val = (await y.data())[0];
  tf.dispose([x, y]);
  return val; // centipawns
}

/**
 * Public API: pick best move using NN + minimax.
 */
export async function pickNNMove(pieces, color, enPassantTarget, depth = 2) {
  const moves = listLegalMoves(pieces, color, enPassantTarget);
  let bestMove = null;
  let bestScore = -Infinity;

  for (const m of moves) {
    const promo = m.needsPromotion && !m.promotionType ? "queen" : m.promotionType;
    const { pieces: after, nextEnPassant } = makeMove(
      pieces,
      m.from,
      m.to,
      promo,
      enPassantTarget
    );

    const score = await minimax(
      after,
      depth - 1,
      false,
      color === "white" ? "black" : "white",
      nextEnPassant
    );

    if (score > bestScore) {
      bestScore = score;
      bestMove = { ...m, promotionType: promo };
    }
  }

  return bestMove;
}

async function minimax(pieces, depth, maximizing, color, enPassantTarget) {
  if (depth === 0) {
    const model = await loadModel();
    const isWhiteTurn = color === "white";
    return await evalPosition(model, pieces, isWhiteTurn);
  }

  const moves = listLegalMoves(pieces, color, enPassantTarget);
  if (moves.length === 0) return maximizing ? -Infinity : Infinity;

  let best = maximizing ? -Infinity : Infinity;
  for (const m of moves) {
    const promo = m.needsPromotion && !m.promotionType ? "queen" : m.promotionType;
    const { pieces: after, nextEnPassant } = makeMove(
      pieces,
      m.from,
      m.to,
      promo,
      enPassantTarget
    );

    const val = await minimax(
      after,
      depth - 1,
      !maximizing,
      maximizing ? "black" : "white",
      nextEnPassant
    );

    if (maximizing) {
      best = Math.max(best, val);
    } else {
      best = Math.min(best, val);
    }
  }

  return best;
}
