// src/rules/chessRules.js

const key = (x, y) => `${x}-${y}`;
const inBounds = (x, y) => x >= 0 && x < 8 && y >= 0 && y < 8;

export const getPiece = (pieces, x, y) => pieces[key(x, y)] || null;

const sameColor = (a, b) => a && b && a.color === b.color;

const pathClear = (pieces, x1, y1, x2, y2) => {
  const dx = Math.sign(x2 - x1);
  const dy = Math.sign(y2 - y1);
  let cx = x1 + dx;
  let cy = y1 + dy;
  while (cx !== x2 || cy !== y2) {
    if (getPiece(pieces, cx, cy)) return false;
    cx += dx;
    cy += dy;
  }
  return true;
};

const rookLike = (pieces, x1, y1, x2, y2) =>
  (x1 === x2 || y1 === y2) && pathClear(pieces, x1, y1, x2, y2);

const bishopLike = (pieces, x1, y1, x2, y2) =>
  Math.abs(x2 - x1) === Math.abs(y2 - y1) && pathClear(pieces, x1, y1, x2, y2);

function findKing(pieces, color) {
  for (let x = 0; x < 8; x++) {
    for (let y = 0; y < 8; y++) {
      const p = getPiece(pieces, x, y);
      if (p && p.type === 'king' && p.color === color) return { x, y };
    }
  }
  return null;
}

export function isSquareAttacked(pieces, x, y, byColor) {
  const pawnRow = x + (byColor === 'white' ? 1 : -1);
  for (const file of [y - 1, y + 1]) {
    const pawn = getPiece(pieces, pawnRow, file);
    if (pawn?.color === byColor && pawn.type === 'pawn') return true;
  }
  for (const [dx, dy] of KNIGHT_STEPS) {
    const knight = getPiece(pieces, x + dx, y + dy);
    if (knight?.color === byColor && knight.type === 'knight') return true;
  }
  for (const [dx, dy] of KING_STEPS) {
    for (let distance = 1; distance < 8; distance += 1) {
      const sx = x + dx * distance;
      const sy = y + dy * distance;
      if (!inBounds(sx, sy)) break;
      const piece = getPiece(pieces, sx, sy);
      if (!piece) continue;
      if (piece.color === byColor && (
        piece.type === 'queen' ||
        piece.type === (dx && dy ? 'bishop' : 'rook') ||
        (distance === 1 && piece.type === 'king')
      )) return true;
      break;
    }
  }
  return false;
}

const KNIGHT_STEPS = [[-2, -1], [-2, 1], [-1, -2], [-1, 2], [1, -2], [1, 2], [2, -1], [2, 1]];
const ROOK_STEPS = [[-1, 0], [0, -1], [0, 1], [1, 0]];
const BISHOP_STEPS = [[-1, -1], [-1, 1], [1, -1], [1, 1]];
const KING_STEPS = [...ROOK_STEPS, ...BISHOP_STEPS];

export function isKingInCheck(pieces, color) {
  const king = findKing(pieces, color);
  if (!king) return false;
  return isSquareAttacked(pieces, king.x, king.y, color === 'white' ? 'black' : 'white');
}

function moveKingForAttackTest(pieces, row, fromY, toY, mover, rookMove = null) {
  const next = { ...pieces };
  delete next[key(row, fromY)];
  next[key(row, toY)] = { ...mover, hasMoved: true };
  if (rookMove) {
    const rook = next[key(row, rookMove.from)];
    delete next[key(row, rookMove.from)];
    if (rook) next[key(row, rookMove.to)] = { ...rook, hasMoved: true };
  }
  return next;
}

function canCastle(pieces, x1, y1, x2, y2, mover) {
  if (mover.type !== 'king') return false;
  if (x1 !== x2 || Math.abs(y2 - y1) !== 2 || mover.hasMoved) return false;

  const color = mover.color;
  const enemy = color === 'white' ? 'black' : 'white';
  const homeRow = color === 'white' ? 7 : 0;
  if (x1 !== homeRow || y1 !== 4) return false;

  const kingSide = y2 > y1;
  const rookY = kingSide ? 7 : 0;
  const rook = getPiece(pieces, homeRow, rookY);
  if (!rook || rook.type !== 'rook' || rook.color !== color || rook.hasMoved) return false;

  const step = kingSide ? 1 : -1;
  for (let cy = y1 + step; cy !== rookY; cy += step) {
    if (getPiece(pieces, homeRow, cy)) return false;
  }

  if (isKingInCheck(pieces, color)) return false;

  const passY = y1 + step;
  const passPieces = moveKingForAttackTest(pieces, homeRow, y1, passY, mover);
  if (isSquareAttacked(passPieces, homeRow, passY, enemy)) return false;

  const rookToY = kingSide ? 5 : 3;
  const finalPieces = moveKingForAttackTest(
    pieces,
    homeRow,
    y1,
    y2,
    mover,
    { from: rookY, to: rookToY },
  );
  if (isSquareAttacked(finalPieces, homeRow, y2, enemy)) return false;

  return true;
}

function isEnPassant(pieces, from, to, enPassantTarget) {
  const mover = getPiece(pieces, from.x, from.y);
  if (!mover || mover.type !== 'pawn' || !enPassantTarget) return false;

  const dx = to.x - from.x;
  const dy = to.y - from.y;
  if (getPiece(pieces, to.x, to.y)) return false;
  const dir = mover.color === 'white' ? -1 : 1;
  return dx === dir && Math.abs(dy) === 1 && enPassantTarget.x === to.x && enPassantTarget.y === to.y;
}

function isPseudoLegalMove(pieces, from, to, isWhiteTurn, enPassantTarget = null) {
  const { x: x1, y: y1 } = from;
  const { x: x2, y: y2 } = to;
  if (!inBounds(x1, y1) || !inBounds(x2, y2) || (x1 === x2 && y1 === y2)) return false;

  const mover = getPiece(pieces, x1, y1);
  if (!mover) return false;
  if ((isWhiteTurn && mover.color !== 'white') || (!isWhiteTurn && mover.color !== 'black')) return false;

  const dest = getPiece(pieces, x2, y2);
  if (sameColor(mover, dest) || dest?.type === 'king') return false;

  const dx = x2 - x1;
  const dy = y2 - y1;
  const adx = Math.abs(dx);
  const ady = Math.abs(dy);

  switch (mover.type) {
    case 'pawn': {
      const dir = mover.color === 'white' ? -1 : 1;
      const startRow = mover.color === 'white' ? 6 : 1;
      if (dx === dir && dy === 0 && !dest) return true;
      if (dx === 2 * dir && dy === 0 && x1 === startRow && !dest && !getPiece(pieces, x1 + dir, y1)) return true;
      if (dx === dir && ady === 1 && dest && dest.color !== mover.color) return true;
      return isEnPassant(pieces, from, to, enPassantTarget);
    }
    case 'rook':
      return rookLike(pieces, x1, y1, x2, y2);
    case 'bishop':
      return bishopLike(pieces, x1, y1, x2, y2);
    case 'queen':
      return rookLike(pieces, x1, y1, x2, y2) || bishopLike(pieces, x1, y1, x2, y2);
    case 'knight':
      return (adx === 1 && ady === 2) || (adx === 2 && ady === 1);
    case 'king':
      if (adx <= 1 && ady <= 1) return true;
      return canCastle(pieces, x1, y1, x2, y2, mover);
    default:
      return false;
  }
}

export function makeMove(pieces, from, to, promotionType = null, enPassantTarget = null) {
  const { x: x1, y: y1 } = from;
  const { x: x2, y: y2 } = to;
  const mover = getPiece(pieces, x1, y1);
  if (!mover) return { pieces, nextEnPassant: null };

  const next = { ...pieces };
  let nextEnPassant = null;
  const movingPiece = { ...mover, hasMoved: true };

  if (movingPiece.type === 'king' && x1 === x2 && Math.abs(y2 - y1) === 2) {
    const kingSide = y2 > y1;
    const rookFromY = kingSide ? 7 : 0;
    const rookToY = kingSide ? 5 : 3;
    const rookFromK = key(x1, rookFromY);
    const rookToK = key(x1, rookToY);
    const rook = next[rookFromK];
    if (rook && rook.type === 'rook' && rook.color === movingPiece.color) {
      next[rookToK] = { ...rook, hasMoved: true };
      delete next[rookFromK];
    }
  }

  if (isEnPassant(pieces, from, to, enPassantTarget)) {
    delete next[key(x1, y2)];
  }

  next[key(x2, y2)] = movingPiece;
  delete next[key(x1, y1)];

  if (movingPiece.type === 'pawn' && Math.abs(x2 - x1) === 2) {
    const dir = movingPiece.color === 'white' ? -1 : 1;
    nextEnPassant = { x: x1 + dir, y: y1 };
  }

  if (movingPiece.type === 'pawn') {
    const promoteRow = movingPiece.color === 'white' ? 0 : 7;
    if (x2 === promoteRow && promotionType) {
      next[key(x2, y2)] = { color: movingPiece.color, type: promotionType, hasMoved: true };
    }
  }

  return { pieces: next, nextEnPassant };
}

export function isLegalMove(pieces, from, to, isWhiteTurn, enPassantTarget = null) {
  if (!isPseudoLegalMove(pieces, from, to, isWhiteTurn, enPassantTarget)) return false;
  const mover = getPiece(pieces, from.x, from.y);
  const color = mover?.color;
  if (!color) return false;
  const { pieces: after } = makeMove(pieces, from, to, null, enPassantTarget);
  return !isKingInCheck(after, color);
}

export function hasAnyLegalMove(pieces, color, enPassantTarget = null) {
  return listLegalMoves(pieces, color, enPassantTarget).length > 0;
}

export function listLegalMoves(pieces, color, enPassantTarget = null) {
  const isWhiteTurn = color === 'white';
  const moves = [];
  for (let x1 = 0; x1 < 8; x1++) {
    for (let y1 = 0; y1 < 8; y1++) {
      const piece = getPiece(pieces, x1, y1);
      if (!piece || piece.color !== color) continue;
      const targets = [];
      if (piece.type === 'pawn') {
        const direction = color === 'white' ? -1 : 1;
        targets.push([x1 + direction, y1], [x1 + direction, y1 - 1], [x1 + direction, y1 + 1]);
        if (x1 === (color === 'white' ? 6 : 1)) targets.push([x1 + 2 * direction, y1]);
      } else {
        const steps = piece.type === 'knight' ? KNIGHT_STEPS
          : piece.type === 'bishop' ? BISHOP_STEPS
            : piece.type === 'rook' ? ROOK_STEPS : KING_STEPS;
        const sliding = ['bishop', 'rook', 'queen'].includes(piece.type);
        for (const [dx, dy] of steps) {
          for (let distance = 1; distance <= (sliding ? 7 : 1); distance += 1) {
            const x2 = x1 + dx * distance;
            const y2 = y1 + dy * distance;
            if (!inBounds(x2, y2)) break;
            targets.push([x2, y2]);
            if (getPiece(pieces, x2, y2)) break;
          }
        }
        if (piece.type === 'king' && !piece.hasMoved) targets.push([x1, y1 - 2], [x1, y1 + 2]);
      }
      for (const [x2, y2] of targets) {
        const from = { x: x1, y: y1 };
        const to = { x: x2, y: y2 };
        if (!isLegalMove(pieces, from, to, isWhiteTurn, enPassantTarget)) continue;

        const target = getPiece(pieces, x2, y2);
        const needsPromotion = piece.type === 'pawn' && (x2 === 0 || x2 === 7);
        const baseMove = { from, to, capture: !!target || isEnPassant(pieces, from, to, enPassantTarget), needsPromotion };
        if (needsPromotion) {
          for (const promotionType of ['queen', 'rook', 'bishop', 'knight']) {
            moves.push({ ...baseMove, promotionType });
          }
        } else {
          moves.push(baseMove);
        }
      }
    }
  }
  return moves;
}
