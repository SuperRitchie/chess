// src/ai/mctsAI.js
import { listLegalMoves, makeMove, isKingInCheck, hasAnyLegalMove } from './chessRules';

/**
 * A compact MCTS (UCT) player for our simple board structure.
 * - Time-limited (default 1200ms) or iteration-limited (default 2000 iters)
 * - Random playouts with shallow cutoff
 * - Auto-queen on promotions
 */

const UCT_C = Math.sqrt(2);

function clone(obj) {
    return JSON.parse(JSON.stringify(obj));
}

function sideEnemy(color) { return color === 'white' ? 'black' : 'white'; }

function isTerminal(pieces, color, enPassantTarget) {
    const inCheck = isKingInCheck(pieces, color);
    const any = hasAnyLegalMove(pieces, color, enPassantTarget);
    if (!any) {
        if (inCheck) return { done: true, result: -1 }; // side-to-move checkmated (loss)
        return { done: true, result: 0 }; // stalemate
    }
    return { done: false, result: 0 };
}

function evaluateMaterial(pieces) {
    const value = { pawn: 1, knight: 3, bishop: 3, rook: 5, queen: 9, king: 0 };
    let score = 0;
    for (const k in pieces) {
        const p = pieces[k];
        const s = value[p.type] ?? 0;
        score += (p.color === 'white') ? s : -s;
    }
    return score;
}

function rollout(pieces, toMove, enPassantTarget, maxPlies = 40) {
    // Random playout capped by plies; if terminal, score exact outcome from root side POV later.
    let color = toMove;
    let ep = enPassantTarget;
    let steps = 0;
    while (steps < maxPlies) {
        const term = isTerminal(pieces, color, ep);
        if (term.done) {
            // +1 win for side-to-move? No—this result is from current side's POV (losing if checkmated)
            // We'll translate at backprop depending on root role.
            return { terminal: true, result: term.result };
        }
        // random move
        const moves = listLegalMoves(pieces, color, ep);
        if (moves.length === 0) {
            // Should have been caught by terminal, but guard anyway
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
    // Heuristic at cutoff: material balance (white positive)
    const mat = evaluateMaterial(pieces);
    // Convert to [-1,1] from “toMove at root is WHITE” assumption later; return raw mat here
    return { terminal: false, mat };
}

class Node {
    constructor(parent, pieces, toMove, enPassantTarget, move = null, rootColor = 'white') {
        this.parent = parent;
        this.children = [];
        this.pieces = pieces;
        this.toMove = toMove;
        this.enPassantTarget = enPassantTarget;
        this.move = move; // move taken from parent to reach this node
        this.rootColor = rootColor;

        this.untriedMoves = listLegalMoves(pieces, toMove, enPassantTarget);
        this.N = 0;
        this.W = 0; // sum of results from root's perspective (win=1, loss=0)
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

    isLeaf() { return this.untriedMoves.length === 0 && this.children.length === 0; }
}

/**
 * Convert a rollout outcome to [0,1] from the ROOT player's perspective.
 * rolloutResult:
 *  - if terminal: result = -1 (loss for side-to-move), 0 (stalemate), or +1 (not used here)
 *  - else: mat heuristic (>0 favors white)
 */
function scoreFromRoot(resultObj, node) {
    // Determine perspective: rootColor vs who the result references.
    // In rollout() terminal case, result is from the current side-to-move perspective at terminal test.
    // We need to translate to the ROOT player's perspective.
    if (resultObj.terminal) {
        // At terminal test, "result = -1" means side-to-move is checkmated.
        // If side-to-move == rootColor at that test, root lost → 0; else root won → 1.
        // We don't know which color was side-to-move at that instant here, so we infer:
        // We started playout from this node's toMove; the terminal check happened before a move by that color, so use this node.toMove.
        const side = node.toMove;
        if (resultObj.result === -1) return side === node.rootColor ? 0 : 1;
        if (resultObj.result === 0) return 0.5; // stalemate
        // (Not produced in our logic, but keep safe)
        return side === node.rootColor ? 1 : 0;
    } else {
        // Heuristic: material >0 favors white. Map to [0,1] via sigmoid-ish clamp.
        const mat = resultObj.mat;
        const favorWhite = 1 / (1 + Math.exp(-0.5 * mat)); // smooth 0..1
        return node.rootColor === 'white' ? favorWhite : (1 - favorWhite);
    }
}

export async function pickMCTSMove(pieces, color, enPassantTarget, {
    timeMs = 1200,
    maxIterations = 2000,
    rolloutDepth = 40
} = {}) {
    const root = new Node(null, clone(pieces), color, enPassantTarget, null, color);

    const tEnd = Date.now() + timeMs;
    let iters = 0;

    while (iters < maxIterations && Date.now() < tEnd) {
        // 1) Selection
        let node = root;
        while (node.untriedMoves.length === 0 && node.children.length > 0) {
            node = node.bestChild();
        }
        // 2) Expansion
        if (node.untriedMoves.length > 0) {
            node = node.expand();
        }
        // 3) Simulation
        let simPieces = clone(node.pieces);
        let simToMove = node.toMove;
        let simEP = node.enPassantTarget;
        const result = rollout(simPieces, simToMove, simEP, rolloutDepth);

        // 4) Backpropagation
        let cur = node;
        const score = scoreFromRoot(result, node); // from ROOT player's perspective
        while (cur) {
            cur.N += 1;
            cur.W += score;
            cur = cur.parent;
        }
        iters++;
    }

    // pick child with highest visit count
    if (root.children.length === 0) return null;
    const best = root.children.reduce((a, b) => (a.N > b.N ? a : b));
    return best.move || null;
}
