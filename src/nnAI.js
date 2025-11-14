// src/nnAI.js
/* global tf */

import { listLegalMoves, makeMove, getPiece } from "./chessRules";

/**
 * Loads model once (singleton).
 * public/nn/model.json is a "layers-model" taking [1, 8, 8, 13].
 */
let _modelPromise = null;

async function loadModel() {
  if (!_modelPromise) {
    const base = process.env.PUBLIC_URL || "";
    const url = `${base}/nn/model.json`;
    console.log("Loading NN model from", url);
    _modelPromise = tf.loadLayersModel(url);
  }
  return _modelPromise;
}

/**
 * Feature extractor matching ml/features.py
 *  - 12 planes: 6 piece types × 2 colors
 *  - 1 plane: side to move
 * Layout: [1, 8, 8, 13] (channels-last)
 */
function featuresFromBoard(pieces, isWhiteTurn) {
  const planes = {
    white: { pawn: [], knight: [], bishop: [], rook: [], queen: [], king: [] },
    black: { pawn: [], knight: [], bishop: [], rook: [], queen: [], king: [] },
  };

  // Build 12 planes: each is 64 entries (one per square)
  for (let x = 0; x < 8; x++) {
    for (let y = 0; y < 8; y++) {
      const p = getPiece(pieces, x, y);

      const onehot = {
        white: { pawn: 0, knight: 0, bishop: 0, rook: 0, queen: 0, king: 0 },
        black: { pawn: 0, knight: 0, bishop: 0, rook: 0, queen: 0, king: 0 },
      };

      if (p) {
        onehot[p.color][p.type] = 1;
      }

      for (const color of ["white", "black"]) {
        for (const t of ["pawn", "knight", "bishop", "rook", "queen", "king"]) {
          planes[color][t].push(onehot[color][t]);
        }
      }
    }
  }

  // Side-to-move plane: same value on all 64 squares
  const sideVal = isWhiteTurn ? 1 : 0;
  const sidePlane = new Array(64).fill(sideVal);

  // Pack into [1, 8, 8, 13] with channel-last layout.
  // Channel order: all whites, then all blacks, then side.
  const data = [];
  for (let x = 0; x < 8; x++) {
    for (let y = 0; y < 8; y++) {
      const idx = x * 8 + y;
      const channels = [
        planes.white.pawn[idx],
        planes.white.knight[idx],
        planes.white.bishop[idx],
        planes.white.rook[idx],
        planes.white.queen[idx],
        planes.white.king[idx],
        planes.black.pawn[idx],
        planes.black.knight[idx],
        planes.black.bishop[idx],
        planes.black.rook[idx],
        planes.black.queen[idx],
        planes.black.king[idx],
        sidePlane[idx],
      ];
      data.push(...channels);
    }
  }

  return tf.tensor(data, [1, 8, 8, 13]);
}

/**
 * Evaluate position: returns centipawns from White's POV.
 */
async function evalPosition(model, pieces, isWhiteTurn) {
  const x = featuresFromBoard(pieces, isWhiteTurn);
  const y = model.predict(x);
  const valArray = await y.data();
  const val = valArray[0];
  tf.dispose([x, y]);
  return val;
}

/**
 * Top-level API used by ChessBoard.js
 * color: "white" or "black"
 */
export async function pickNNMove(
  pieces,
  color,
  enPassantTarget,
  depth = 2
) {
  const moves = listLegalMoves(pieces, color, enPassantTarget);
  let bestMove = null;
  let bestScore = -Infinity;

  for (const m of moves) {
    const promo =
      m.needsPromotion && !m.promotionType ? "queen" : m.promotionType;
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
  if (moves.length === 0) {
    return maximizing ? -Infinity : Infinity;
  }

  let best = maximizing ? -Infinity : Infinity;

  for (const m of moves) {
    const promo =
      m.needsPromotion && !m.promotionType ? "queen" : m.promotionType;
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
