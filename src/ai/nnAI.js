import * as tf from '@tensorflow/tfjs';
import { listLegalMoves, makeMove, getPiece, isKingInCheck } from '../rules/chessRules';
import {
  rankMoves,
  stabilizePolicyValue,
  tacticalSearch,
} from './chessHeuristics';

export const POLICY_CHANNELS = 5;
export const POLICY_SIZE = 64 * 64 * POLICY_CHANNELS;
const LEGACY_POLICY_SIZE = 64 * 64;

let modelPromise = null;
let modelWarmPromise = null;

async function loadModel() {
  if (!modelPromise) {
    modelPromise = tf.loadLayersModel(`${process.env.PUBLIC_URL}/nn/model.json`).catch((error) => {
      modelPromise = null;
      throw error;
    });
  }
  return modelPromise;
}

export async function warmNNModel() {
  if (!modelWarmPromise) {
    modelWarmPromise = loadModel().then(async (model) => {
      const planeCount = Number(model.inputs?.[0]?.shape?.[3]) || 18;
      const input = tf.zeros([1, 8, 8, planeCount]);
      const prediction = model.predict(input);
      const outputs = Array.isArray(prediction) ? prediction : [prediction];

      try {
        await Promise.all(outputs.map((output) => output.data()));
      } finally {
        tf.dispose([input, ...outputs]);
      }
      return model;
    }).catch((error) => {
      modelWarmPromise = null;
      throw error;
    });
  }
  return modelWarmPromise;
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

function writeBoardFeatures(buf, batchIndex, pieces, isWhiteTurn, enPassantTarget, planeCount) {
  const typeToIdx = { pawn: 0, knight: 1, bishop: 2, rook: 3, queen: 4, king: 5 };

  for (let y = 0; y < 8; y++) {
    for (let x = 0; x < 8; x++) {
      const piece = getPiece(pieces, y, x);
      if (piece) {
        const base = piece.color === 'white' ? 0 : 6;
        buf.set(1, batchIndex, y, x, base + typeToIdx[piece.type]);
      }
      buf.set(isWhiteTurn ? 1 : 0, batchIndex, y, x, 12);
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
        for (let x = 0; x < 8; x++) buf.set(1, batchIndex, y, x, 13 + offset);
      }
    });
    if (enPassantTarget) {
      buf.set(1, batchIndex, enPassantTarget.x, enPassantTarget.y, 17);
    }
  }
}

function featuresFromPositions(positions, planeCount = 18) {
  const buf = tf.buffer([positions.length, 8, 8, planeCount], 'float32');
  positions.forEach((position, index) => {
    writeBoardFeatures(
      buf,
      index,
      position.pieces,
      position.color === 'white',
      position.enPassantTarget,
      planeCount,
    );
  });

  return buf.toTensor();
}

async function rawPredictions(model, positions) {
  const planeCount = Number(model.inputs?.[0]?.shape?.[3]) || 18;
  const x = featuresFromPositions(positions, planeCount);
  const prediction = model.predict(x);
  const outputs = Array.isArray(prediction) ? prediction : [null, prediction];
  const [policyTensor, valueTensor] = outputs.length === 2 ? outputs : [null, outputs[0]];

  try {
    const policyData = policyTensor ? await policyTensor.data() : null;
    const valueData = await valueTensor.data();
    const policySize = policyTensor ? Number(policyTensor.shape[policyTensor.shape.length - 1]) : 0;
    return positions.map((_, index) => ({
      policyData: policyData?.subarray(index * policySize, (index + 1) * policySize) || null,
      value: valueData[index],
    }));
  } finally {
    tf.dispose([x, ...outputs.filter(Boolean)]);
  }
}

function uniformPrediction(legalMoves) {
  const uniform = 1 / Math.max(1, legalMoves.length);
  const priors = new Map();
  legalMoves.forEach((move) => priors.set(moveKey(move), uniform));
  return { value: 0, priors, legalMoves, neuralAvailable: false };
}

function predictionForLegalMoves(legalMoves, raw) {
  if (!raw.policyData) return { ...uniformPrediction(legalMoves), value: raw.value };
  const logits = legalMoves.map((move) => raw.policyData[moveToPolicyIndex(move, raw.policyData.length)] ?? -1000000);
  const probabilities = stableSoftmax(logits);
  const priors = new Map();
  legalMoves.forEach((move, index) => priors.set(moveKey(move), probabilities[index]));
  return { value: raw.value, priors, legalMoves, neuralAvailable: true };
}

export async function predictPolicyValueBatchForPositions(positions) {
  const results = new Array(positions.length);
  const pending = [];

  positions.forEach((position, index) => {
    const legalMoves = listLegalMoves(position.pieces, position.color, position.enPassantTarget);
    if (legalMoves.length === 0) {
      results[index] = {
        value: isKingInCheck(position.pieces, position.color) ? -1 : 0,
        priors: new Map(),
        legalMoves,
      };
    } else {
      pending.push({ ...position, index, legalMoves });
    }
  });

  if (pending.length === 0) return results;

  try {
    const model = await warmNNModel();
    const predictions = await rawPredictions(model, pending);
    pending.forEach((position, pendingIndex) => {
      results[position.index] = predictionForLegalMoves(position.legalMoves, predictions[pendingIndex]);
    });
  } catch (error) {
    console.warn('Neural model unavailable; using uniform policy/value fallback.', error);
    pending.forEach((position) => {
      results[position.index] = uniformPrediction(position.legalMoves);
    });
  }

  return results;
}

export async function predictPolicyValueForMoves(pieces, color, enPassantTarget) {
  const [prediction] = await predictPolicyValueBatchForPositions([{ pieces, color, enPassantTarget }]);
  return prediction;
}

function applyMove(pieces, move, enPassantTarget) {
  const promotion = move.promotionType || (move.needsPromotion ? 'queen' : null);
  const result = makeMove(pieces, move.from, move.to, promotion, enPassantTarget);
  return {
    ...result,
    move: { ...move, promotionType: promotion },
  };
}

export async function pickNNMove(
  pieces,
  color,
  enPassantTarget,
  depth = 2,
  {
    predictBatch = predictPolicyValueBatchForPositions,
    rootMoveLimit = 14,
    replyLimit = 12,
    tacticalCandidates = 6,
    tacticalDepth = 2,
    tacticalTimeMs = 700,
    tacticalWeight = 0.45,
  } = {},
) {
  const [rawRoot] = await predictBatch([{ pieces, color, enPassantTarget }]);
  const rootPrediction = stabilizePolicyValue(
    pieces,
    color,
    enPassantTarget,
    rawRoot,
  );
  const rootMoves = rankMoves(rootPrediction).slice(0, Math.max(1, rootMoveLimit));
  if (rootMoves.length === 0) return null;
  if (depth <= 1) return applyMove(pieces, rootMoves[0], enPassantTarget).move;

  const nextColor = color === 'white' ? 'black' : 'white';
  const candidates = rootMoves.map((move) => ({
    ...applyMove(pieces, move, enPassantTarget),
    score: Infinity,
  }));
  const rawReplies = await predictBatch(candidates.map((candidate) => ({
    pieces: candidate.pieces,
    color: nextColor,
    enPassantTarget: candidate.nextEnPassant,
  })));
  const leaves = [];

  candidates.forEach((candidate, candidateIndex) => {
    const replyPrediction = stabilizePolicyValue(
      candidate.pieces,
      nextColor,
      candidate.nextEnPassant,
      rawReplies[candidateIndex],
    );
    if (replyPrediction.legalMoves.length === 0) {
      candidate.score = isKingInCheck(candidate.pieces, nextColor) ? 1 : 0;
      return;
    }

    const replies = rankMoves(replyPrediction).slice(0, Math.max(1, replyLimit));
    replies.forEach((reply) => {
      const afterReply = applyMove(candidate.pieces, reply, candidate.nextEnPassant);
      leaves.push({
        candidateIndex,
        pieces: afterReply.pieces,
        color,
        enPassantTarget: afterReply.nextEnPassant,
      });
    });
  });

  if (leaves.length > 0) {
    const rawLeaves = await predictBatch(leaves);
    leaves.forEach((leaf, leafIndex) => {
      const leafPrediction = stabilizePolicyValue(
        leaf.pieces,
        leaf.color,
        leaf.enPassantTarget,
        rawLeaves[leafIndex],
      );
      candidates[leaf.candidateIndex].score = Math.min(
        candidates[leaf.candidateIndex].score,
        leafPrediction.value,
      );
    });
  }

  const scoredCandidates = candidates.map((candidate) => {
    const moveId = moveKey(candidate.move);
    const priorBonus = 0.1 * (rootPrediction.priors.get(moveId) || 0);
    const positionalBonus = 0.04 * (rootPrediction.moveScores.get(moveId) || 0);
    return { candidate, score: candidate.score + priorBonus + positionalBonus };
  }).sort((first, second) => second.score - first.score);

  const tacticalDeadline = Date.now() + Math.max(1, tacticalTimeMs);
  const tacticalCache = new Map();
  const verifyCount = Math.min(Math.max(1, tacticalCandidates), scoredCandidates.length);
  for (let index = 0; index < verifyCount; index += 1) {
    const item = scoredCandidates[index];
    const tacticalScore = -tacticalSearch(
      item.candidate.pieces,
      nextColor,
      item.candidate.nextEnPassant,
      {
        depth: tacticalDepth,
        extensionBudget: 1,
        maxMoves: 6,
        maxNodes: 700,
        deadline: tacticalDeadline,
        cache: tacticalCache,
      },
    );
    item.score = (1 - tacticalWeight) * item.score + tacticalWeight * tacticalScore;
  }

  const best = scoredCandidates.reduce(
    (current, item) => (!current || item.score > current.score ? item : current),
    null,
  );

  return best?.candidate.move || null;
}
