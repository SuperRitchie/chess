// src/ai/nnAI.js
import * as tf from "@tensorflow/tfjs";
import { listLegalMoves, makeMove, getPiece } from "./chessRules";

/**
 * Loads model once (singleton).
 */
let _modelPromise = null;
async function loadModel() {
  if (!_modelPromise) {
    _modelPromise = tf.loadGraphModel(process.env.PUBLIC_URL + "/nn/model.json");
  }
  return _modelPromise;
}

/**
 * Feature extractor (JS) must match Python features in ml/features.py.
 * We'll use 12 piece planes (one per piece type per color) + side-to-move plane.
 * Output shape: [8,8,13] -> flatten to [8*8*13].
 */
function featuresFromBoard(pieces, isWhiteTurn) {
  const planes = {
    white: { pawn: [], knight: [], bishop: [], rook: [], queen: [], king: [] },
    black: { pawn: [], knight: [], bishop: [], rook: [], queen: [], king: [] },
  };
  for (let x = 0; x < 8; x++) {
    for (let y = 0; y < 8; y++) {
      const p = getPiece(pieces, x, y);
      const onehot = {
        white: { pawn:0, knight:0, bishop:0, rook:0, queen:0, king:0 },
        black: { pawn:0, knight:0, bishop:0, rook:0, queen:0, king:0 },
      };
      if (p) onehot[p.color][p.type] = 1;
      Object.keys(planes).forEach(color => {
        Object.keys(planes[color]).forEach(t => planes[color][t].push(onehot[color][t]));
      });
    }
  }
  const side = new Array(64).fill(isWhiteTurn ? 1 : 0);

  // concat 12 piece planes + 1 side plane
  const all = []
    .concat(planes.white.pawn, planes.white.knight, planes.white.bishop, planes.white.rook, planes.white.queen, planes.white.king)
    .concat(planes.black.pawn, planes.black.knight, planes.black.bishop, planes.black.rook, planes.black.queen, planes.black.king)
    .flat()
    .concat(side);

  return tf.tensor(all, [1, 8 * 8 * 13]); // batch of 1
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
export async function pickNNMove(pieces, color, enPassantTarget) {
  const model = await loadModel();
  const isWhite = color === "white";
  const moves = listLegalMoves(pieces, color, enPassantTarget);
  if (moves.length === 0) return null;

  let best = null;
  let bestScore = -Infinity;

  for (const m of moves) {
    const promo = m.needsPromotion && !m.promotionType ? "queen" : m.promotionType;
    const { pieces: after } = makeMove(pieces, m.from, m.to, promo, enPassantTarget);
    const score = await evalPosition(model, after, !isWhite); // next to move is opponent
    // White likes large +, Black likes large - (mirror by multiplying)
    const signed = isWhite ? score : -score;
    if (signed > bestScore) { bestScore = signed; best = { ...m, promotionType: promo }; }
  }
  return best;
}
