// src/ai/nnAI.js
import * as tf from "@tensorflow/tfjs";
import { listLegalMoves, makeMove, getPiece } from "./chessRules";

/**
 * Loads model once (singleton).
 */
let _modelPromise = null;
async function loadModel() {
  if (!_modelPromise) {
    _modelPromise = tf.loadLayersModel(process.env.PUBLIC_URL + "/nn/model.json");
  }
  return _modelPromise;
}

/**
 * Feature extractor (JS) must match Python features in ml/features.py.
 * We'll use 12 piece planes (one per piece type per color) + side-to-move plane.
 * Output shape: [8,8,13] -> 4d tensor
 */
function featuresFromBoard(pieces, isWhiteTurn) {
  // tensor shape: [1, 8, 8, 13] (channels-last)
  const buf = tf.buffer([1, 8, 8, 13], "float32");

  const typeToIdx = { pawn: 0, knight: 1, bishop: 2, rook: 3, queen: 4, king: 5 };

  for (let y = 0; y < 8; y++) {
    for (let x = 0; x < 8; x++) {
      const p = getPiece(pieces, x, y);
      if (p) {
        const base = (p.color === "white") ? 0 : 6;     // white: 0..5, black: 6..11
        const c = base + typeToIdx[p.type];
        buf.set(1, 0, y, x, c);
      }
      // side-to-move plane (channel 12)
      buf.set(isWhiteTurn ? 1 : 0, 0, y, x, 12);
    }
  }

  return buf.toTensor();
}

/**
 * Score: predicted centipawns from White's perspective.
 */
async function evalPosition(model, pieces, isWhiteTurn) {
  const x = featuresFromBoard(pieces, isWhiteTurn);
  const y = model.predict(x);
  const val = (await y.data())[0];
  tf.dispose([x, y]);
  return val; // centipawns (float)
}

/**
 * Pick best move by 1-ply lookahead using the NN eval.
 * If promotion required and not specified, auto-queen.
 */
export async function pickNNMove(pieces, color, enPassantTarget, depth = 2) {
  const moves = listLegalMoves(pieces, color, enPassantTarget);
  let bestMove = null;
  let bestScore = -Infinity;
  for (const m of moves) {
    const promo = m.needsPromotion && !m.promotionType ? 'queen' : m.promotionType;
    const { pieces: after, nextEnPassant } = makeMove(pieces, m.from, m.to, promo, enPassantTarget);
    // depth-1 because we already made one move
    const nextColor = (color === "white") ? "black" : "white";
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
    const isWhiteTurn = (color === "white");
    return await evalPosition(model, pieces, isWhiteTurn);
  }

  const moves = listLegalMoves(pieces, color, enPassantTarget);
  if (moves.length === 0) {
    // checkmate or stalemate, need to refer back 
    return (color === "white") ? -Infinity : Infinity;
  }

  const maximizing = (color === "white"); // maximize White's eval
  let best = maximizing ? -Infinity : Infinity;

  const nextColor = (color === "white") ? "black" : "white";

  for (const m of moves) {
    const promo = m.needsPromotion && !m.promotionType ? "queen" : m.promotionType;
    const { pieces: after, nextEnPassant } = makeMove(pieces, m.from, m.to, promo, enPassantTarget);

    const val = await minimax(after, depth - 1, nextColor, nextEnPassant);

    best = maximizing ? Math.max(best, val) : Math.min(best, val);
  }

  return best;
}

