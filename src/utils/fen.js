// src/utils/fen.js
// Minimal FEN from your board representation { "x-y": {color,type,hasMoved} }
// Files are columns (y), ranks are rows (x). White moves "up" (x decreasing).

import { getPiece } from "../chessRules";

const pieceCode = {
  white: { king:'K', queen:'Q', rook:'R', bishop:'B', knight:'N', pawn:'P' },
  black: { king:'k', queen:'q', rook:'r', bishop:'b', knight:'n', pawn:'p' },
};

export function boardToFEN(pieces, isWhiteTurn, castling = "-", ep = null, halfmove = 0, fullmove = 1) {
  const rows = [];
  for (let x = 0; x < 8; x++) {
    let row = "";
    let empties = 0;
    for (let y = 0; y < 8; y++) {
      const p = getPiece(pieces, x, y);
      if (!p) {
        empties++;
      } else {
        if (empties > 0) { row += String(empties); empties = 0; }
        row += pieceCode[p.color][p.type];
      }
    }
    if (empties > 0) row += String(empties);
    rows.push(row);
  }
  // FEN ranks must go 8->1, but our x goes 0..7 from top to bottom, so join as-is:
  const placement = rows.join("/");
  const turn = isWhiteTurn ? "w" : "b";
  const epSquare = ep ? algebraic(ep.x, ep.y) : "-";
  return `${placement} ${turn} ${castling} ${epSquare} ${halfmove} ${fullmove}`;
}

function algebraic(x, y) {
  const files = "abcdefgh";
  return files[y] + (8 - x);
}
