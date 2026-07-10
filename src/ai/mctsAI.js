import { makeMove, isKingInCheck, listLegalMoves } from '../rules/chessRules';
import { moveKey, predictPolicyValueForMoves } from './nnAI';

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
  }

  get meanValue() {
    return this.visitCount === 0 ? 0 : this.valueSum / this.visitCount;
  }

  selectChild(cpuct) {
    let best = null;
    let bestScore = -Infinity;
    const parentVisits = Math.max(1, this.visitCount);

    for (const child of this.children) {
      const q = child.visitCount === 0 ? 0 : -child.meanValue;
      const u = cpuct * child.prior * Math.sqrt(parentVisits) / (1 + child.visitCount);
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

async function evaluateAndExpand(node) {
  const terminal = terminalValue(node.pieces, node.toMove, node.enPassantTarget);
  if (terminal !== null) return terminal;

  let prediction;
  try {
    prediction = await predictPolicyValueForMoves(
      node.pieces,
      node.toMove,
      node.enPassantTarget,
    );
  } catch (error) {
    console.warn('Neural MCTS inference failed; using uniform priors.', error);
  }

  const { value, priors, legalMoves } = prediction || uniformPrediction(
    node.pieces,
    node.toMove,
    node.enPassantTarget,
  );
  node.expand(legalMoves, priors);
  return Number.isFinite(value) ? value : 0;
}

export async function pickMCTSMove(
  pieces,
  color,
  enPassantTarget,
  { timeMs = 1200, maxIterations = 3000, cpuct = DEFAULT_CPUCT } = {},
) {
  const root = new Node(null, clonePieces(pieces), color, enPassantTarget);
  const rootValue = await evaluateAndExpand(root);
  root.visitCount = 1;
  root.valueSum = rootValue;
  if (root.children.length === 0) return null;

  const deadline = Date.now() + Math.max(1, timeMs);
  let iterations = 0;
  while (iterations < maxIterations && Date.now() < deadline) {
    let node = root;
    while (node.children.length > 0) {
      node = node.selectChild(cpuct);
      if (!node) break;
    }
    if (!node) break;

    const value = await evaluateAndExpand(node);
    node.backup(value);
    iterations += 1;
  }

  const best = root.children.reduce((current, child) => {
    if (!current) return child;
    if (child.visitCount !== current.visitCount) {
      return child.visitCount > current.visitCount ? child : current;
    }
    return child.meanValue < current.meanValue ? child : current;
  }, null);

  return best?.move || null;
}
