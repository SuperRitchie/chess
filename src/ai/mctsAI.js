import { listLegalMoves, makeMove, isKingInCheck, hasAnyLegalMove } from '../rules/chessRules';

const UCT_C = Math.sqrt(2);

function clone(obj) {
  return JSON.parse(JSON.stringify(obj));
}

function sideEnemy(color) {
  return color === 'white' ? 'black' : 'white';
}

function isTerminal(pieces, color, enPassantTarget) {
  const inCheck = isKingInCheck(pieces, color);
  const any = hasAnyLegalMove(pieces, color, enPassantTarget);
  if (!any) {
    if (inCheck) return { done: true, result: -1 };
    return { done: true, result: 0 };
  }
  return { done: false, result: 0 };
}

function evaluateMaterial(pieces) {
  const value = { pawn: 1, knight: 3, bishop: 3, rook: 5, queen: 9, king: 0 };
  let score = 0;
  for (const k in pieces) {
    const p = pieces[k];
    const s = value[p.type] ?? 0;
    score += p.color === 'white' ? s : -s;
  }
  return score;
}

function rollout(pieces, toMove, enPassantTarget, maxPlies = 40) {
  let color = toMove;
  let ep = enPassantTarget;
  let steps = 0;
  while (steps < maxPlies) {
    const term = isTerminal(pieces, color, ep);
    if (term.done) {
      return { terminal: true, result: term.result };
    }

    const moves = listLegalMoves(pieces, color, ep);
    if (moves.length === 0) {
      return { terminal: true, result: 0 };
    }

    const m = moves[Math.floor(Math.random() * moves.length)];
    if (m.needsPromotion && !m.promotionType) m.promotionType = 'queen';
    const { pieces: next, nextEnPassant } = makeMove(pieces, m.from, m.to, m.promotionType, ep);
    pieces = next;
    ep = nextEnPassant;
    color = sideEnemy(color);
    steps++;
  }

  const mat = evaluateMaterial(pieces);
  return { terminal: false, mat };
}

class Node {
  constructor(parent, pieces, toMove, enPassantTarget, move = null, rootColor = 'white') {
    this.parent = parent;
    this.children = [];
    this.pieces = pieces;
    this.toMove = toMove;
    this.enPassantTarget = enPassantTarget;
    this.move = move;
    this.rootColor = rootColor;

    this.untriedMoves = listLegalMoves(pieces, toMove, enPassantTarget);
    this.N = 0;
    this.W = 0;
  }

  uctScore() {
    if (this.N === 0) return Infinity;
    const mean = this.W / this.N;
    const parentN = this.parent ? Math.max(1, this.parent.N) : 1;
    return mean + UCT_C * Math.sqrt(Math.log(parentN) / this.N);
  }

  bestChild() {
    return this.children.reduce((a, b) => (a.uctScore() > b.uctScore() ? a : b));
  }

  expand() {
    if (this.untriedMoves.length === 0) return this;
    const m = this.untriedMoves.pop();
    const promo = m.needsPromotion && !m.promotionType ? 'queen' : m.promotionType;
    const { pieces: next, nextEnPassant } = makeMove(this.pieces, m.from, m.to, promo, this.enPassantTarget);
    const child = new Node(this, next, sideEnemy(this.toMove), nextEnPassant, { ...m, promotionType: promo }, this.rootColor);
    this.children.push(child);
    return child;
  }
}

function scoreFromRoot(resultObj, node) {
  if (resultObj.terminal) {
    const side = node.toMove;
    if (resultObj.result === -1) return side === node.rootColor ? 0 : 1;
    if (resultObj.result === 0) return 0.5;
    return side === node.rootColor ? 1 : 0;
  }

  const mat = resultObj.mat;
  const favorWhite = 1 / (1 + Math.exp(-0.5 * mat));
  return node.rootColor === 'white' ? favorWhite : 1 - favorWhite;
}

export async function pickMCTSMove(
  pieces,
  color,
  enPassantTarget,
  { timeMs = 1200, maxIterations = 2000, rolloutDepth = 40 } = {},
) {
  const root = new Node(null, clone(pieces), color, enPassantTarget, null, color);

  const tEnd = Date.now() + timeMs;
  let iters = 0;

  while (iters < maxIterations && Date.now() < tEnd) {
    let node = root;
    while (node.untriedMoves.length === 0 && node.children.length > 0) {
      node = node.bestChild();
    }

    if (node.untriedMoves.length > 0) {
      node = node.expand();
    }

    const simPieces = clone(node.pieces);
    const simToMove = node.toMove;
    const simEP = node.enPassantTarget;
    const result = rollout(simPieces, simToMove, simEP, rolloutDepth);

    let cur = node;
    const score = scoreFromRoot(result, node);
    while (cur) {
      cur.N += 1;
      cur.W += score;
      cur = cur.parent;
    }
    iters++;
  }

  if (root.children.length === 0) return null;
  const best = root.children.reduce((a, b) => (a.N > b.N ? a : b));
  return best.move || null;
}
