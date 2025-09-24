// src/chessRules.js

const key = (x, y) => `${x}-${y}`;
const inBounds = (x, y) => x >= 0 && x < 8 && y >= 0 && y < 8;

export const getPiece = (pieces, x, y) => pieces[key(x, y)] || null;

const sameColor = (a, b) => a && b && a.color === b.color;

const pathClear = (pieces, x1, y1, x2, y2) => {
  const dx = Math.sign(x2 - x1);
  const dy = Math.sign(y2 - y1);
  let cx = x1 + dx, cy = y1 + dy;
  while (cx !== x2 || cy !== y2) {
    if (getPiece(pieces, cx, cy)) return false;
    cx += dx; cy += dy;
  }
  return true;
};

const rookLike = (pieces, x1, y1, x2, y2) =>
  (x1 === x2 || y1 === y2) && pathClear(pieces, x1, y1, x2, y2);

const bishopLike = (pieces, x1, y1, x2, y2) =>
  Math.abs(x2 - x1) === Math.abs(y2 - y1) && pathClear(pieces, x1, y1, x2, y2);

/** Locate king of a color */
function findKing(pieces, color) {
  for (let x = 0; x < 8; x++) {
    for (let y = 0; y < 8; y++) {
      const p = getPiece(pieces, x, y);
      if (p && p.type === 'king' && p.color === color) return { x, y };
    }
  }
  return null;
}

/** Is square (x,y) attacked by any piece of byColor? (pseudo-legal attacks) */
export function isSquareAttacked(pieces, x, y, byColor) {
  for (let sx = 0; sx < 8; sx++) {
    for (let sy = 0; sy < 8; sy++) {
      const p = getPiece(pieces, sx, sy);
      if (!p || p.color !== byColor) continue;

      const dx = x - sx, dy = y - sy;
      const adx = Math.abs(dx), ady = Math.abs(dy);

      switch (p.type) {
        case 'pawn': {
          const dir = (byColor === 'white') ? -1 : 1;
          if (dx === dir && Math.abs(dy) === 1) return true;
          break;
        }
        case 'knight':
          if ((adx === 1 && ady === 2) || (adx === 2 && ady === 1)) return true;
          break;
        case 'bishop':
          if (Math.abs(dx) === Math.abs(dy) && pathClear(pieces, sx, sy, x, y)) return true;
          break;
        case 'rook':
          if ((sx === x || sy === y) && pathClear(pieces, sx, sy, x, y)) return true;
          break;
        case 'queen':
          if (((sx === x || sy === y) || (Math.abs(dx) === Math.abs(dy))) &&
            pathClear(pieces, sx, sy, x, y)) return true;
          break;
        case 'king':
          if (adx <= 1 && ady <= 1) return true;
          break;
        default:
          break;
      }
    }
  }
  return false;
}

/** Is king of `color` currently in check? */
export function isKingInCheck(pieces, color) {
  const king = findKing(pieces, color);
  if (!king) return false;
  const enemy = (color === 'white') ? 'black' : 'white';
  return isSquareAttacked(pieces, king.x, king.y, enemy);
}

/** CASTLING: check if a king move (x1,y1) -> (x2,y2) is a valid castle for mover.color */
function canCastle(pieces, x1, y1, x2, y2, mover) {
  // Must be a king trying to move two squares horizontally
  if (mover.type !== 'king') return false;
  if (x1 !== x2 || Math.abs(y2 - y1) !== 2) return false;
  if (mover.hasMoved) return false;

  const color = mover.color;
  const enemy = color === 'white' ? 'black' : 'white';
  const homeRow = (color === 'white') ? 7 : 0;
  if (x1 !== homeRow) return false;

  const kingSide = (y2 > y1); // right
  const rookY = kingSide ? 7 : 0;
  const rook = getPiece(pieces, homeRow, rookY);
  if (!rook || rook.type !== 'rook' || rook.color !== color || rook.hasMoved) return false;

  // Squares between king and rook must be empty
  const step = kingSide ? 1 : -1;
  for (let cy = y1 + step; cy !== rookY; cy += step) {
    if (getPiece(pieces, homeRow, cy)) return false;
  }

  // King may not be in check, and squares he crosses/lands on must not be attacked
  if (isKingInCheck(pieces, color)) return false;
  const pass1Y = y1 + step;
  const destY = y1 + 2 * step;

  if (isSquareAttacked(pieces, homeRow, pass1Y, enemy)) return false;
  if (isSquareAttacked(pieces, homeRow, destY, enemy)) return false;

  return true;
}

/** Core movement rules (no king-safety yet) + turn check + CASTLING shape */
function isPseudoLegalMove(pieces, from, to, isWhiteTurn) {
  const { x: x1, y: y1 } = from;
  const { x: x2, y: y2 } = to;

  if (!inBounds(x1, y1) || !inBounds(x2, y2)) return false;
  if (x1 === x2 && y1 === y2) return false;

  const mover = getPiece(pieces, x1, y1);
  if (!mover) return false;

  // turn check
  if ((isWhiteTurn && mover.color !== 'white') || (!isWhiteTurn && mover.color !== 'black')) {
    return false;
  }

  const dest = getPiece(pieces, x2, y2);
  if (sameColor(mover, dest)) return false; // can’t capture own piece

  const dx = x2 - x1, dy = y2 - y1;
  const adx = Math.abs(dx), ady = Math.abs(dy);

  switch (mover.type) {
    case 'pawn': {
      const dir = mover.color === 'white' ? -1 : 1;   // white moves toward smaller x
      const startRow = mover.color === 'white' ? 6 : 1;

      // forward empty
      if (dx === dir && dy === 0 && !dest) return true;

      // double from start (both squares empty)
      if (
        dx === 2 * dir &&
        dy === 0 &&
        x1 === startRow &&
        !dest &&
        !getPiece(pieces, x1 + dir, y1)
      ) return true;

      // diagonal capture
      if (dx === dir && Math.abs(dy) === 1 && dest && dest.color !== mover.color) return true;

      return false;
    }
    case 'rook':
      return rookLike(pieces, x1, y1, x2, y2);
    case 'bishop':
      return bishopLike(pieces, x1, y1, x2, y2);
    case 'queen':
      return rookLike(pieces, x1, y1, x2, y2) || bishopLike(pieces, x1, y1, x2, y2);
    case 'knight':
      return (adx === 1 && ady === 2) || (adx === 2 && ady === 1);
    case 'king': {
      // normal king step
      if (adx <= 1 && ady <= 1) return true;
      // castling
      return canCastle(pieces, x1, y1, x2, y2, mover);
    }
    default:
      return false;
  }
}

/** Simulate and return new board. Supports: normal moves, castling, promotionType */
export function makeMove(pieces, from, to, promotionType = null) {
  const { x: x1, y: y1 } = from;
  const { x: x2, y: y2 } = to;
  const mover = getPiece(pieces, x1, y1);
  if (!mover) return pieces;

  const next = { ...pieces };
  const movingPiece = { ...mover, hasMoved: true };

  // Handle castling rook shift if king castles
  if (movingPiece.type === 'king' && x1 === x2 && Math.abs(y2 - y1) === 2) {
    const kingSide = (y2 > y1);
    const rookFromY = kingSide ? 7 : 0;
    const rookToY = kingSide ? 5 : 3;
    const rookFromK = key(x1, rookFromY);
    const rookToK = key(x1, rookToY);
    const rook = next[rookFromK];
    if (rook && rook.type === 'rook' && rook.color === movingPiece.color) {
      // move rook
      next[rookToK] = { ...rook, hasMoved: true };
      delete next[rookFromK];
    }
  }

  // Move the piece
  next[key(x2, y2)] = movingPiece;
  delete next[key(x1, y1)];

  // Promotion
  if (movingPiece.type === 'pawn') {
    const promoteRow = movingPiece.color === 'white' ? 0 : 7;
    if (x2 === promoteRow && promotionType) {
      next[key(x2, y2)] = { color: movingPiece.color, type: promotionType, hasMoved: true };
    }
  }

  return next;
}

/** Full legality: obeys movement AND cannot leave your own king in check */
export function isLegalMove(pieces, from, to, isWhiteTurn) {
  if (!isPseudoLegalMove(pieces, from, to, isWhiteTurn)) return false;

  const mover = getPiece(pieces, from.x, from.y);
  const color = mover?.color;
  if (!color) return false;

  // simulate and verify king safety
  const after = makeMove(pieces, from, to);
  return !isKingInCheck(after, color);
}

/** Does side `color` have at least one legal move? */
export function hasAnyLegalMove(pieces, color) {
  const isWhiteTurn = (color === 'white');
  for (let x1 = 0; x1 < 8; x1++) {
    for (let y1 = 0; y1 < 8; y1++) {
      const p = getPiece(pieces, x1, y1);
      if (!p || p.color !== color) continue;

      for (let x2 = 0; x2 < 8; x2++) {
        for (let y2 = 0; y2 < 8; y2++) {
          if (isLegalMove(pieces, { x: x1, y: y1 }, { x: x2, y: y2 }, isWhiteTurn)) {
            return true;
          }
        }
      }
    }
  }
  return false;
}
