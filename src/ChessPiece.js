import React from 'react';
import './ChessPiece.css';

import white_pawn from './assets/white-pawn.svg';
import white_rook from './assets/white-rook.svg';
import white_knight from './assets/white-knight.svg';
import white_bishop from './assets/white-bishop.svg';
import white_queen from './assets/white-queen.svg';
import white_king from './assets/white-king.svg';

import black_pawn from './assets/black-pawn.svg';
import black_rook from './assets/black-rook.svg';
import black_knight from './assets/black-knight.svg';
import black_bishop from './assets/black-bishop.svg';
import black_queen from './assets/black-queen.svg';
import black_king from './assets/black-king.svg';

// map piece name to image
const pieceImages = {
    'white-pawn': white_pawn,
    'white-rook': white_rook,
    'white-knight': white_knight,
    'white-bishop': white_bishop,
    'white-queen': white_queen,
    'white-king': white_king,
    'black-pawn': black_pawn,
    'black-rook': black_rook,
    'black-knight': black_knight,
    'black-bishop': black_bishop,
    'black-queen': black_queen,
    'black-king': black_king,
};

// Create a ChessPiece component
// svg chess pieces from https://commons.wikimedia.org/wiki/Category:SVG_chess_pieces



function ChessPiece({ piece }) {
    if (!piece) return null;
    const pieceName = piece ? `${piece.color}-${piece.type}` : '';
    const pieceSrc = pieceImages[`${piece.color}-${piece.type}`];
    return (
        <div className="chess-piece">
            {piece && <img src={pieceSrc} alt={pieceName} />}
        </div>
    );
}

export default ChessPiece;
