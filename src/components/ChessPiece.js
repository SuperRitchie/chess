import React from 'react';
import './ChessPiece.css';

import whitePawn from '../assets/white-pawn.svg';
import whiteRook from '../assets/white-rook.svg';
import whiteKnight from '../assets/white-knight.svg';
import whiteBishop from '../assets/white-bishop.svg';
import whiteQueen from '../assets/white-queen.svg';
import whiteKing from '../assets/white-king.svg';

import blackPawn from '../assets/black-pawn.svg';
import blackRook from '../assets/black-rook.svg';
import blackKnight from '../assets/black-knight.svg';
import blackBishop from '../assets/black-bishop.svg';
import blackQueen from '../assets/black-queen.svg';
import blackKing from '../assets/black-king.svg';

const pieceImages = {
  'white-pawn': whitePawn,
  'white-rook': whiteRook,
  'white-knight': whiteKnight,
  'white-bishop': whiteBishop,
  'white-queen': whiteQueen,
  'white-king': whiteKing,
  'black-pawn': blackPawn,
  'black-rook': blackRook,
  'black-knight': blackKnight,
  'black-bishop': blackBishop,
  'black-queen': blackQueen,
  'black-king': blackKing,
};

function ChessPiece({ piece }) {
  if (!piece) return null;

  const pieceName = `${piece.color}-${piece.type}`;
  const pieceSrc = pieceImages[pieceName];

  return (
    <div className="chess-piece">
      <img src={pieceSrc} alt={pieceName} />
    </div>
  );
}

export default ChessPiece;
