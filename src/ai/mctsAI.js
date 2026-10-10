import { makeMove, isKingInCheck, listLegalMoves } from '../rules/chessRules';
import {
  moveKey,
  predictPolicyValueBatchForPositions,
  predictPolicyValueForMoves,
} from './nnAI';
import {
  positionKey,
  stabilizePolicyValue,
} from './chessHeuristics';
import { findMateInOne, searchTacticalMoves } from './tacticalSearch';

const DEFAULT_CPUCT = 1.5;

function clonePieces(pieces) {
  return JSON.parse(JSON.stringify(pieces));
}

function opponent(color) {
  return color === 'white' ? 'black' : 'white';
}

function terminalValue(pieces, color, enPassantTarget) {
  const moves = listLegalMoves(pieces, color, enPassantTarget);
  if (moves.length > 0) return null;
  return isKingInCheck(pieces, color) ? -1 : 0;
}

function uniformPrediction(pieces, color, enPassantTarget) {
  const legalMoves = listLegalMoves(pieces, color, enPassantTarget);
  const probability = 1 / Math.max(1, legalMoves.length);
  return {
    value: 0,
    legalMoves,
    priors: new Map(legalMoves.map((move) => [moveKey(move), probability])),
    neuralAvailable: false,
  };
}

class Node {
  constructor(parent, pieces, toMove, enPassantTarget, move = null, prior = 0) {
    this.parent = parent;
    this.pieces = pieces;
    this.toMove = toMove;
    this.enPassantTarget = enPassantTarget;
    this.move = move;
    this.prior = prior;
    this.children = [];
    this.visitCount = 0;
    this.valueSum = 0;
    this.virtualVisits = 0;
  }

  get meanValue() {
    return this.visitCount === 0 ? 0 : this.valueSum / this.visitCount;
  }

  selectChild(cpuct) {
    let best = null;
    let bestScore = -Infinity;
    const parentVisits = Math.max(1, this.visitCount + this.virtualVisits);

    for (const child of this.children) {
      const effectiveVisits = child.visitCount + child.virtualVisits;
      const q = effectiveVisits === 0 ? 0 : (-child.valueSum - child.virtualVisits) / effectiveVisits;
      const u = cpuct * child.prior * Math.sqrt(parentVisits) / (1 + effectiveVisits);
      const score = q + u;
      if (score > bestScore) {
        bestScore = score;
        best = child;
      }
    }
    return best;
  }

  expand(legalMoves, priors) {
    if (this.children.length > 0) return;
    for (const move of legalMoves) {
      const promotion = move.promotionType || (move.needsPromotion ? 'queen' : null);
      const { pieces: nextPieces, nextEnPassant } = makeMove(
        this.pieces,
        move.from,
        move.to,
        promotion,
        this.enPassantTarget,
      );
      this.children.push(
        new Node(
          this,
          nextPieces,
          opponent(this.toMove),
          nextEnPassant,
          { ...move, promotionType: promotion },
          priors.get(moveKey(move)) || 0,
        ),
      );
    }
  }

  backup(value) {
    let node = this;
    let currentValue = value;
    while (node) {
      node.visitCount += 1;
      node.valueSum += currentValue;
      currentValue = -currentValue;
      node = node.parent;
    }
  }
}

function adjustVirtualVisits(node, amount) {
  let current = node;
  while (current) {
    current.virtualVisits += amount;
    current = current.parent;
  }
}

async function evaluateAndExpand(node, predictionCache) {
  const [value] = await evaluateAndExpandBatch([node], predictionCache);
  return value;
}

async function evaluateAndExpandBatch(nodes, predictionCache = new Map()) {
  const values = new Array(nodes.length);
  const pendingByKey = new Map();

  nodes.forEach((node, index) => {
    const terminal = terminalValue(node.pieces, node.toMove, node.enPassantTarget);
    if (terminal === null) {
      const key = positionKey(node.pieces, node.toMove, node.enPassantTarget);
      const cached = predictionCache.get(key);
      if (cached) {
        node.expand(cached.legalMoves, cached.priors);
        values[index] = Number.isFinite(cached.value) ? cached.value : 0;
      } else {
        if (!pendingByKey.has(key)) pendingByKey.set(key, []);
        pendingByKey.get(key).push({ node, index });
      }
    } else {
      values[index] = terminal;
    }
  });

  const groups = [...pendingByKey.entries()];
  if (groups.length === 0) return values;
  const representatives = groups.map(([, items]) => items[0].node);

  let predictions;
  try {
    if (typeof predictPolicyValueBatchForPositions !== 'function') throw new Error('batch inference unavailable');
    predictions = await predictPolicyValueBatchForPositions(
      representatives.map((node) => ({
        pieces: node.pieces,
        color: node.toMove,
        enPassantTarget: node.enPassantTarget,
      })),
    );
  } catch (error) {
    predictions = await Promise.all(
      representatives.map((node) => predictPolicyValueForMoves(
        node.pieces,
        node.toMove,
        node.enPassantTarget,
      ).catch(() => uniformPrediction(node.pieces, node.toMove, node.enPassantTarget))),
    );
  }

  groups.forEach(([key, items], predictionIndex) => {
    const node = items[0].node;
    const prediction = stabilizePolicyValue(
      node.pieces,
      node.toMove,
      node.enPassantTarget,
      predictions[predictionIndex] || uniformPrediction(
        node.pieces,
        node.toMove,
        node.enPassantTarget,
      ),
    );
    predictionCache.set(key, prediction);
    items.forEach(({ node: pendingNode, index }) => {
      pendingNode.expand(prediction.legalMoves, prediction.priors);
      values[index] = Number.isFinite(prediction.value) ? prediction.value : 0;
    });
  });

  return values;
}

export async function pickMCTSMove(
  pieces,
  color,
  enPassantTarget,
  {
    timeMs = 2000,
    maxIterations = 4096,
    batchSize = 16,
    cpuct = DEFAULT_CPUCT,
    tacticalDepth = 3,
    tacticalReserveMs = 1000,
    onSearchComplete,
  } = {},
) {
  const predictionCache = new Map();
  const rootPieces = clonePieces(pieces);
  const legalRootMoves = listLegalMoves(rootPieces, color, enPassantTarget);
  if (legalRootMoves.length === 0) return null;
  const mate = findMateInOne(rootPieces, color, enPassantTarget, legalRootMoves);
  if (mate) return mate;

  const root = new Node(null, rootPieces, color, enPassantTarget);
  const rootValue = await evaluateAndExpand(root, predictionCache);
  root.visitCount = 1;
  root.valueSum = rootValue;
  if (root.children.length === 0) return null;

  const deadline = Date.now() + Math.max(1, timeMs);
  const searchDeadline = deadline - Math.min(Math.max(0, tacticalReserveMs), timeMs / 2);
  let iterations = 0;
  while (iterations < maxIterations && Date.now() < searchDeadline) {
    const selected = [];
    const currentBatchSize = Math.min(Math.max(1, batchSize), maxIterations - iterations);

    for (let index = 0; index < currentBatchSize; index++) {
      let node = root;
      while (node.children.length > 0) {
        node = node.selectChild(cpuct);
        if (!node) break;
      }
      if (!node || selected.includes(node)) break;
      adjustVirtualVisits(node, 1);
      selected.push(node);
    }
    if (selected.length === 0) break;

    const values = await evaluateAndExpandBatch(selected, predictionCache);
    selected.forEach((node, index) => {
      adjustVirtualVisits(node, -1);
      node.backup(values[index]);
    });
    iterations += selected.length;
  }

  const neuralScores = new Map(root.children.map((child) => [
    moveKey(child.move),
    (child.visitCount === 0 ? rootValue : -child.meanValue) + 0.15 * child.prior,
  ]));
  const result = searchTacticalMoves(rootPieces, color, enPassantTarget, {
    moves: root.children.map((child) => child.move),
    neuralScores,
    timeMs: Math.max(1, deadline - Date.now()),
    maxDepth: tacticalDepth,
  });
  onSearchComplete?.({ ...result, iterations, rootVisits: root.visitCount });
  return result.move;
}
