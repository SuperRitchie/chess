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

export function isLegalMove(pieces, from, to, isWhiteTurn) {
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

      // forward empty?
      if (dx === dir && dy === 0 && !dest) return true;

      // double from start (both squares must be empty)
      if (
        dx === 2 * dir &&
        dy === 0 &&
        x1 === startRow &&
        !dest &&
        !getPiece(pieces, x1 + dir, y1)
      ) return true;

      // diagonal capture
      if (dx === dir && Math.abs(dy) === 1 && dest && dest.color !== mover.color) return true;

      return false; // (no en passant/promo here)
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
      return adx <= 1 && ady <= 1; // castling not implemented here
    default:
      return false;
  }
}

export function makeMove(pieces, from, to) {
  const { x: x1, y: y1 } = from;
  const { x: x2, y: y2 } = to;
  const mover = getPiece(pieces, x1, y1);
  if (!mover) return pieces;

  const next = { ...pieces };
  next[`${x2}-${y2}`] = mover;
  delete next[`${x1}-${y1}`];

  // handle promotion/castling/en-passant
  return next;
}
